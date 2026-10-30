import os
import numpy as np
import pandas as pd
import torch
import csv
import ast
import functools
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

import pickle

from utils.backbones import load_extractor

from dotenv import load_dotenv

load_dotenv()

AUDIOS_FOLDER = os.getenv("AUDIOS_FOLDER")
EMBEDDINGS_FOLDER = os.getenv("EMBEDDINGS_FOLDER")
TRIPLETS_FOLDER = os.getenv("TRIPLETS_FOLDER")

#threads used to read the audio embeddings. There is one small file per track, so the time is spent waiting for the disk
NUM_THREADS = 32

def load_audio_embeddings(song, return_local_tokens, backbone):
    '''
    Reads and normalizes the embeddings of one track, from {EMBEDDINGS_FOLDER}/{backbone}/{global or local}/{song_id}.npy
    (the song id starts with the folder of the audio dataset, like original/00/1076600).
    Returns (song_id, (global_, local_)), where local_ is None when return_local_tokens is False.
    '''
    song_id = song.split('.')[0]
    global_path = os.path.join(EMBEDDINGS_FOLDER, backbone, 'global', f"{song_id}.npy")

    global_ = torch.from_numpy(np.load(global_path)).float().squeeze()
    # normalize once
    global_ = F.normalize(global_, dim=-1)

    local_ = None
    if return_local_tokens:
        local_path = os.path.join(EMBEDDINGS_FOLDER, backbone, 'local', f"{song_id}.npy")
        local_ = torch.from_numpy(np.load(local_path)).float().squeeze()
        local_ = F.normalize(local_, dim=-1)

    return song_id, (global_, local_)

def cache_items(items_list, dataset, split, return_local_tokens, cache_type, purpose, backbone):
    '''
    items_list: list of items to cache (either text or audio)
    dataset: str, either 'mtsq' or 'mtiq'
    return_local_tokens: bool, whether to return local tokens from the dataset
    cache_type: str, either 'audio' or 'text'
    backbone: str, key of BACKBONE_DIMS. Its embeddings are read (audio) or computed (text), and its caches have their own folder
    '''

    return_local_str = "local" if return_local_tokens else "no_local"

    #pre loading all text embeddings
    cache_dict = {}

    #unique_texts = triplets_df['modification_text'].unique()

    file_name = f"cache/{backbone}/{cache_type}_{purpose}_cache_{dataset}_{split}_{return_local_str}.pt"

    if os.path.exists(file_name):
        print(f"Loading pre-computed {file_name} cache...")
        cache_dict = torch.load(file_name, map_location="cpu")
    else:
        if cache_type == "text":
            clap_model = load_extractor(backbone)
            if return_local_tokens:
                for i, text in enumerate(items_list):
                    print(f"Pre-loading text embeddings: {i+1}/{len(items_list)}", end='\r')

                    g, l = clap_model.get_text_tokens(text)
                    g = g.detach().cpu()
                    l = l.detach().cpu()
                    
                    l = F.normalize(l.float().squeeze(), dim=-1)
                    g = F.normalize(g.float().squeeze(), dim=-1)

                    cache_dict[text] = (g, l)
            else:
                for i, text in enumerate(items_list):
                    print(f"Pre-loading text embeddings: {i+1}/{len(items_list)}", end='\r')

                    g, _ = clap_model.get_text_tokens(text)
                    g = g.detach().cpu()
                    g = F.normalize(g.float().squeeze(), dim=-1)
                    
                    l = None

                    cache_dict[text] = (g, l)
            del clap_model

        elif cache_type == "audio":
            #the tracks are read by several threads at once. executor.map returns the results in the order of items_list
            load = functools.partial(load_audio_embeddings, return_local_tokens=return_local_tokens, backbone=backbone)
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                for i, (song_id, embeddings) in enumerate(executor.map(load, items_list)):
                    if i % 1000 == 0 or i + 1 == len(items_list):
                        print(f"Pre-loading audio embeddings: {i+1}/{len(items_list)}", end='\r')

                    cache_dict[song_id] = embeddings

        #save cache
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(cache_dict, file_name)

    torch.cuda.empty_cache()

    return cache_dict


