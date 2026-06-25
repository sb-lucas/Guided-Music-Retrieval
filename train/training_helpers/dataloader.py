import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import pandas as pd
import numpy as np
import torch.nn.functional as F
import librosa
import sys, os
import random
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../feature_extractors')))
from clap import HuggingFaceClap

import pickle

class TripletDataset(Dataset):
    def __init__(self, triplets_path, return_local_tokens, split='train', dataset='controlled'):

        clap_model = HuggingFaceClap()

        self.local_audio_tokens_folder = "path/to/clap_local"
        self.global_audio_tokens_folder = "path/to/clap_global"
        self.triplets_df = pd.read_csv(triplets_path)
        
        self.return_local_tokens = return_local_tokens

        self.all_songs = pd.concat([self.triplets_df["reference_song"], self.triplets_df["target_song"]]).unique()

        #pre loading all audio embeddings
        self.audio_cache = {}
        if os.path.exists(f"cache/audio_cache_{dataset}_{split}.pt"):
            print("Loading pre-computed audio cache...")
            self.audio_cache = torch.load(f"cache/audio_cache_{dataset}_{split}.pt", map_location="cpu")
        else:
            for i, song in enumerate(self.all_songs):
                print(f"Pre-loading audio embeddings: {i+1}/{len(self.all_songs)}. song = {song}: ", end='\r')
                song_id = song.split('.')[0]
                local_path = os.path.join(self.local_audio_tokens_folder, f"{song_id}.npy")
                global_path = os.path.join(self.global_audio_tokens_folder, f"{song_id}.npy")

                local = torch.from_numpy(np.load(local_path)).float().squeeze()
                global_ = torch.from_numpy(np.load(global_path)).float().squeeze()

                # normalize once
                local = F.normalize(local, dim=-1)
                global_ = F.normalize(global_, dim=-1)

                self.audio_cache[song_id] = (local, global_)

            #save cache
            torch.save(self.audio_cache, f"cache/audio_cache_{dataset}_{split}.pt")

        #pre loading all text embeddings
        self.text_cache = {}
        if os.path.exists(f"cache/text_cache_{dataset}_{split}.pt"):
            print("Loading pre-computed text cache...")
            self.text_cache = torch.load(f"cache/text_cache_{dataset}_{split}.pt", map_location="cpu")
        else:
            unique_texts = self.triplets_df['modification_text'].unique()
            for text in unique_texts:
                g, l = clap_model.get_text_tokens(text)
                g = g.detach().cpu()
                l = l.detach().cpu()
                
                l = F.normalize(l.float().squeeze(), dim=-1)
                g = F.normalize(g.float().squeeze(), dim=-1)

                self.text_cache[text] = (g, l)
            #save cache
            torch.save(self.text_cache, f"cache/text_cache_{dataset}_{split}.pt")

        del clap_model
        torch.cuda.empty_cache()  # optional but safe

        print('TripletDataset initialized:')

    def get_num_songs(self):
        return len(self.all_songs)

    def get_audio_len(self):
        sample_row = self.triplets_df.iloc[0]
        sample_song = sample_row['reference_song']
        sample_song_path = os.path.join("path/to/audios", sample_song)
        audio, sr = librosa.load(sample_song_path, sr=None)

        return len(audio)/sr #return length in seconds

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        row = self.triplets_df.iloc[idx]

        reference_song = row['reference_song'].split('.')[0]
        target_song = row['target_song'].split('.')[0]
        
        ref_local, ref_global = self.audio_cache[reference_song]
        tar_local, tar_global = self.audio_cache[target_song]

        modification_text = row['modification_text']
        mod_global, mod_local = self.text_cache[modification_text]

        if self.return_local_tokens:
            return ref_local, ref_global, mod_local, mod_global, tar_local, tar_global

        return ref_global, mod_global, tar_global

class GeometricBalancedSampler(BatchSampler):
    def __init__(self, triplets_df, batch_size, seed=42):
        self.triplets_df = triplets_df.reset_index(drop=True)
        self.batch_size = batch_size
        
        self.rng = random.Random(seed)

        # Group indices by modification type 
        self.mod_groups = self.triplets_df.groupby('modification').groups
        
        # Get the median value of the class counts for geometric mean calculation
        class_counts = self.triplets_df['modification'].value_counts()
        self.median_val = class_counts.median()
        
        # Max multiplication for upsampling minority classes
        self.max_multiplier = 5

        # Pre-calculate target sizes for each modification type
        self.target_sizes = {}
        for mod, count in class_counts.items():
            geo_mean = math.floor(math.sqrt(count * self.median_val))
            safe_max = int(count * self.max_multiplier)
            self.target_sizes[mod] = min(geo_mean, safe_max)
        
    def __iter__(self):
        # Generate the initial queue based on the mode (train/val)
        epoch_indices = self.epoch_resampler()

        # Simple, linear batching without collision tracking
        batch = []
        for idx in epoch_indices:
            batch.append(idx)
            
            # Yield full batch
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Yield leftovers, but strictly drop if the leftover size is exactly 1
        # This prevents the Contrastive Loss math from collapsing to 0.000
        if len(batch) > 1:
            yield batch

    def __len__(self):
        # The epoch size is the sum of all pre-calculated target sizes
        total_samples = sum(self.target_sizes.values())
        # Using floor division inherently accounts for dropping incomplete remainders
        return total_samples // self.batch_size
    
    def epoch_resampler(self):
        # GEOMETRIC DOWNSAMPLING / UPSAMPLING
        epoch_indices = []
        for mod, indices in self.mod_groups.items():
            idx_list = list(indices)
            target = self.target_sizes[mod]
            
            if len(idx_list) >= target:
                # Downsample majority classes
                sampled = self.rng.sample(idx_list, target)
            else:
                # Upsample minority classes WITH replacement
                sampled = self.rng.choices(idx_list, k=target)
                
            epoch_indices.extend(sampled)
            
        # Shuffling ensures that the balanced classes are evenly distributed across batches
        self.rng.shuffle(epoch_indices)
        return epoch_indices