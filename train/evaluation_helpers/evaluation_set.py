import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.caching import cache_items

from dotenv import load_dotenv

load_dotenv()

TRIPLETS_FOLDER = os.getenv("TRIPLETS_FOLDER")

class EvaluationSet():
    def __init__(self, dataset_name, split, return_local_tokens, backbone):
        self.dataset_name = dataset_name.upper()     #name of the dataset folder, 'MTSQ' or 'MTIQ'
        self.split = split
        self.return_local_tokens = return_local_tokens
        self.backbone = backbone                     #key of BACKBONE_DIMS, the embeddings it extracted are the ones loaded

        if self.dataset_name not in ("MTSQ", "MTIQ"):
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_folder = f'{TRIPLETS_FOLDER}/{self.dataset_name}'
        self.triplets_df = pd.read_csv(f'{dataset_folder}/{self.split}_triplets.csv')

        #---------------------------------- the pool of tracks to retrieve from -------------------------------------#
        #every track of the split in splits.csv, even the ones that were not used in any triplet
        splits_df = pd.read_csv(f'{dataset_folder}/splits.csv')
        self.all_songs = splits_df[splits_df['split'] == self.split]['song'].tolist()

        if len(self.all_songs) == 0:
            raise ValueError(f"No tracks of the split '{self.split}' in {dataset_folder}/splits.csv")

        #positions in the pool. They depend only on the dataset, the methods return their embeddings in this order
        self.song_ids = [song.split('.')[0] for song in self.all_songs]
        self.song_to_position = {song_id: position for position, song_id in enumerate(self.song_ids)}
        self.position_to_song = dict(enumerate(self.song_ids))

        #position in the pool of the reference and of the target of each triplet, tensors of shape [number of triplets]
        self.reference_positions = self.get_positions(self.triplets_df['reference_song'])
        self.target_positions = self.get_positions(self.triplets_df['target_song'])

        #---------------------------------- pre loading the embeddings -------------------------------------#
        dataset = self.dataset_name.lower()

        #pre loading all audio embeddings
        self.audio_cache = cache_items(self.all_songs, dataset, self.split, self.return_local_tokens, cache_type='audio', purpose='evaluation', backbone=self.backbone)
        self.check_cache(self.audio_cache, self.song_ids, 'audio')

        #pre loading all text embeddings
        modification_texts = self.triplets_df['modification_text'].unique()
        self.text_cache = cache_items(modification_texts, dataset, self.split, self.return_local_tokens, cache_type='text', purpose='evaluation', backbone=self.backbone)
        self.check_cache(self.text_cache, modification_texts, 'text')

    def __len__(self):
        return len(self.triplets_df)

    def get_queries(self, start, end):
        """
        The inputs of the triplets start:end, in a dict that every method reads the keys it needs from:
            reference_songs, modification_texts: lists of strings, as in the triplets
            ref_global, mod_text_global: [B, D] global embeddings of the reference and of the modification text
        and, when the set returns local tokens:
            ref_local, mod_text_local: [B, L, D] local tokens, padded with zeros to the longest of the batch
            ref_len, mod_text_len: [B] number of tokens before the padding
        """
        rows = self.triplets_df.iloc[start:end]
        reference_ids = rows['reference_song'].str.split('.').str[0].tolist()
        modification_texts = rows['modification_text'].tolist()

        queries = {
            'reference_songs': rows['reference_song'].tolist(),
            'modification_texts': modification_texts,
            'ref_global': torch.stack([self.audio_cache[song_id][0] for song_id in reference_ids]),
            'mod_text_global': torch.stack([self.text_cache[text][0] for text in modification_texts]),
        }

        if self.return_local_tokens:
            ref_local = [self.audio_cache[song_id][1] for song_id in reference_ids]
            mod_text_local = [self.text_cache[text][1] for text in modification_texts]

            queries['ref_len'] = torch.tensor([tokens.size(0) for tokens in ref_local])
            queries['mod_text_len'] = torch.tensor([tokens.size(0) for tokens in mod_text_local])
            queries['ref_local'] = pad_sequence(ref_local, batch_first=True)
            queries['mod_text_local'] = pad_sequence(mod_text_local, batch_first=True)

        return queries

    def get_positions(self, songs):
        #positions in the pool of a column of songs of the triplets
        positions = songs.str.split('.').str[0].map(self.song_to_position)

        if positions.isna().any():
            raise ValueError(f"{positions.isna().sum()} tracks of the triplets are not in the pool of the split '{self.split}'")

        return torch.tensor(positions.values, dtype=torch.long)

    def check_cache(self, cache, keys, cache_type):
        #the cache files are named by dataset and split, so a file built for a different list of items would be loaded silently
        missing = [key for key in keys if key not in cache]
        if missing:
            raise ValueError(f"The {cache_type} cache of {self.dataset_name}/{self.split} lacks {len(missing)} items, for example {missing[0]}. "
                             f"Delete cache/{cache_type}_cache_{self.dataset_name.lower()}_{self.split}_* so it is rebuilt")
