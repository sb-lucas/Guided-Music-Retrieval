import pandas as pd
import os
import librosa as rosa
from transformers import AutoProcessor, ClapModel
import torch
import torch.nn.functional as F
import numpy as np
import gc
import multiprocessing as mp
from multiprocessing import Pool
import random

WEIGHTS_PATH = 'path/to/essentia/weights'
AUDIOS_FOLDER = 'path/to/dataset/audios'
EMBEDDINGS_GLOBAL_FOLDER = 'path/to/dataset/extracted_features/clap_global'
EMBEDDINGS_LOCAL_FOLDER = 'path/to/dataset/extracted_features/clap_local'

class CLAP():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    def get_audio_embeddings(self, audio_path):
        audio_data, _ = rosa.load(audio_path, sr=48000)  # shape (T,)
        audio_data = audio_data.reshape(1, -1)  # (1, T)

        input_text = [""]
        inputs = self.processor(text=input_text, audios=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        #local tokens, of shape [L,D], where L is the number of tokens and D is the dimension of the embeddings
        audio_tokens = outputs.audio_model_output.last_hidden_state  # [1, 1024, 2, 32]
        audio_tokens = audio_tokens.squeeze(0)  # [1024, 2, 32]
        audio_tokens = audio_tokens.permute(1, 2, 0)  # [2, 32, 1024]
        emb_size = audio_tokens.shape[-1]  # 1024
        local_audio_tokens = audio_tokens.reshape(-1, emb_size)  # [64, 1024]

        global_audio_tokens = outputs.audio_embeds

        return global_audio_tokens, local_audio_tokens
    
def process_many_files(task):
    dataset, files = task
    model = CLAP()

    #make sure the dataset folder exists in the embeddings folder
    dataset_global_save_path = os.path.join(EMBEDDINGS_GLOBAL_FOLDER, dataset)
    dataset_local_save_path = os.path.join(EMBEDDINGS_LOCAL_FOLDER, dataset)

    os.makedirs(dataset_global_save_path, exist_ok=True)
    os.makedirs(dataset_local_save_path, exist_ok=True)

    #define file extension according to dataset
    if dataset == 'mood_change':
        file_extension = '.wav'
    else:
        file_extension = '.mp3'

    #iterate over the files and extract embeddings
    for i, audio_file in enumerate(files):
        print(f"\nProcessing row {i+1}/{len(files)}", flush=True)
        
        subdir_global_save_path = os.path.join(dataset_global_save_path, os.path.dirname(audio_file))
        subdir_local_save_path = os.path.join(dataset_local_save_path, os.path.dirname(audio_file))
        os.makedirs(subdir_global_save_path, exist_ok=True)
        os.makedirs(subdir_local_save_path, exist_ok=True)

        global_embedding_save_path = os.path.join(subdir_global_save_path, os.path.basename(audio_file) + '.npy')
        local_embedding_save_path = os.path.join(subdir_local_save_path, os.path.basename(audio_file) + '.npy')

        if os.path.exists(global_embedding_save_path) and os.path.exists(local_embedding_save_path):
            print(f"Embedding already exists for {audio_file}, skipping.", flush=True)
            continue

        global_embedding, local_embedding = model.get_audio_embeddings(os.path.join(AUDIOS_FOLDER, dataset, audio_file + file_extension))

        print(f"Saving embedding for {audio_file} to {global_embedding_save_path}", flush=True)
        
        np.save(global_embedding_save_path, global_embedding.detach().cpu().numpy())
        np.save(local_embedding_save_path, local_embedding.detach().cpu().numpy())

        #cleanup
        del global_embedding, local_embedding
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

    datasets = ['mood_change', 'slowed', 'add_instrument', 'remove_instrument', 'original']

    print(f"Starting processing of datasets: {datasets}\n")

    for dataset in datasets:
        dataset_path = os.path.join(AUDIOS_FOLDER, dataset)

        #get all audio files in the dataset        
        audio_files = []
        for subfolder in os.listdir(dataset_path):
            subfolder_path = os.path.join(dataset_path, subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith(('.wav', '.mp3')):
                    audio_files.append(os.path.join(subfolder, file.split('.')[0]))

        audio_files = sorted(audio_files)

        NUM_WORKERS = 8
        splits = split_list(audio_files, NUM_WORKERS)

        tasks = [(dataset, chunk) for chunk in splits]

        print(f"Dataset {dataset} has {len(audio_files)} audio files, split into {NUM_WORKERS} chunks.\n")

        with Pool(processes=NUM_WORKERS) as pool:
           results = pool.map(process_many_files, tasks)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()