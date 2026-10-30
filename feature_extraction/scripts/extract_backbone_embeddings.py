import os
import gc
import sys
import random
import argparse
import importlib
import functools
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../extractors')))
from dotenv import load_dotenv

load_dotenv()

AUDIOS_FOLDER = os.getenv("AUDIOS_FOLDER")
EMBEDDINGS_FOLDER = os.getenv("EMBEDDINGS_FOLDER")

#module and class of the extractor of each backbone, imported only when used. The name is also the folder of its embeddings:
#{EMBEDDINGS_FOLDER}/{backbone}/{global or local}/{dataset}/{audio_file}.npy
EXTRACTORS = {
    'hf_clap': ('hf_clap', 'HF_CLAP'),
    'laion_clap': ('laion_clap_extractor', 'LAION_CLAP'),
    'musicnn': ('musicnn', 'MusiCNN'),
    #'muq': ('muq_extractor', 'MUQ'),
}

DATASETS = ['mood_change', 'slowed', 'add_instrument', 'remove_instrument', 'original']
NUM_WORKERS = 8

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Extraction")
    parser.add_argument('--backbone', type=str, required=True, choices=list(EXTRACTORS), help='Backbone that extracts the embeddings')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS, choices=DATASETS, help='Audio datasets to process')
    parser.add_argument('--tokens', type=str, nargs='+', default=['global', 'local'], choices=['global', 'local'], help='Which embeddings to save')

    return parser.parse_args()

def process_many_files(task, backbone, token_types):
    dataset, files = task
    module, class_name = EXTRACTORS[backbone]
    model = getattr(importlib.import_module(module), class_name)()

    #backbones without local tokens (MusiCNN) only save the global ones
    has_local = getattr(model, 'has_local_tokens', True)
    if not has_local:
        token_types = [token_type for token_type in token_types if token_type != 'local']

    #define file extension according to dataset
    if dataset == 'mood_change':
        file_extension = '.wav'
    else:
        file_extension = '.mp3'

    #iterate over the files and extract embeddings
    for i, audio_file in enumerate(files):
        print(f"\nProcessing row {i+1}/{len(files)}", flush=True)

        #one file for each type of token
        save_paths = {token_type: os.path.join(EMBEDDINGS_FOLDER, backbone, token_type, dataset, audio_file + '.npy') for token_type in token_types}

        if all(os.path.exists(path) for path in save_paths.values()):
            print(f"Embedding already exists for {audio_file}, skipping.", flush=True)
            continue

        global_embedding, local_embedding = model.get_audio_tokens(os.path.join(AUDIOS_FOLDER, dataset, audio_file + file_extension))
        embeddings = {'global': global_embedding, 'local': local_embedding}

        for token_type, path in save_paths.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, embeddings[token_type].detach().cpu().numpy())

        print(f"Saved {audio_file}: global {tuple(global_embedding.shape)}" + (f", local {tuple(local_embedding.shape)}" if has_local else ""), flush=True)

        #cleanup
        del global_embedding, local_embedding, embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return f"Finished processing audio_files for dataset {dataset}"

def split_list(lst, Y):
    random.seed(42)  # for reproducibility
    random.shuffle(lst)
    N = len(lst)
    k, m = divmod(N, Y)
    return [
        lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]
        for i in range(Y)
    ]

def main():
    args = parse_args()
    print(f"Starting {args.backbone} on datasets: {args.datasets}, tokens: {args.tokens}\n")

    process = functools.partial(process_many_files, backbone=args.backbone, token_types=args.tokens)

    for dataset in args.datasets:
        dataset_path = os.path.join(AUDIOS_FOLDER, dataset)

        #get all audio files in the dataset
        audio_files = []
        for subfolder in os.listdir(dataset_path):
            subfolder_path = os.path.join(dataset_path, subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(subfolder, file.split('.')[0]))

        audio_files = sorted(audio_files)

        splits = split_list(audio_files, NUM_WORKERS)
        tasks = [(dataset, chunk) for chunk in splits]

        print(f"Dataset {dataset} has {len(audio_files)} audio files, split into {NUM_WORKERS} chunks.\n")

        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(process, tasks)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
