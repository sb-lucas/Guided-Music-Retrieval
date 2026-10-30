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
import pickle

from utils.caching import cache_items

from dotenv import load_dotenv

load_dotenv()

AUDIOS_FOLDER = os.getenv("AUDIOS_FOLDER")

class TripletDataset(Dataset):
    def __init__(self, triplets_path, return_local_tokens, backbone, split='train', dataset='controlled'):
        self.backbone = backbone
        self.audios_folder = AUDIOS_FOLDER
        self.triplets_df = pd.read_csv(triplets_path)

        if split == 'val':
            #shuffle it once to ungroup the data, but keep it deterministic for reproducibility
            self.triplets_df = self.triplets_df.sample(frac=1, random_state=0).reset_index(drop=True)
        
        self.return_local_tokens = return_local_tokens

        #pre loading all audio embeddings
        self.all_songs = pd.concat([self.triplets_df["reference_song"], self.triplets_df["target_song"]]).unique()
        self.audio_cache = cache_items(self.all_songs, dataset, split, self.return_local_tokens, cache_type='audio', purpose='training', backbone=self.backbone)

        #pre loading all text embeddings
        self.text_cache = cache_items(self.triplets_df['modification_text'].unique(), dataset, split, self.return_local_tokens, cache_type='text', purpose='training', backbone=self.backbone)

        #create query and target ids for the loss
        query_key = self.triplets_df['reference_song'] + '||' + self.triplets_df['modification']
        self.query_ids  = pd.factorize(query_key)[0]
        self.target_ids = pd.factorize(self.triplets_df['target_song'].str.split('.').str[0])[0]

        print('TripletDataset initialized:')

    def get_num_songs(self):
        return len(self.all_songs)

    def get_audio_len(self):
        sample_row = self.triplets_df.iloc[0]
        sample_song = sample_row['reference_song']
        sample_song_path = os.path.join(self.audios_folder, sample_song)
        audio, sr = librosa.load(sample_song_path, sr=None)

        return len(audio)/sr #return length in seconds

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        row = self.triplets_df.iloc[idx]

        reference_song = row['reference_song'].split('.')[0]
        target_song = row['target_song'].split('.')[0]
        
        ref_global, ref_local = self.audio_cache[reference_song]
        tar_global, tar_local = self.audio_cache[target_song]

        modification_text = row['modification_text']
        mod_global, mod_local = self.text_cache[modification_text]

        query_id = self.query_ids[idx]
        target_id = self.target_ids[idx]

        if self.return_local_tokens:
            return ref_local, ref_global, mod_local, mod_global, tar_local, tar_global, query_id, target_id

        return ref_global, mod_global, tar_global, query_id, target_id

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