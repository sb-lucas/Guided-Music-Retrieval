import pandas as pd
import os
import librosa
from transformers import AutoProcessor, ClapModel
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch.nn.functional as F
from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN, TensorflowPredictMAEST
from pathlib import Path
import essentia.standard as es
import numpy as np
from essentia import Pool as EssentiaPool
import gc
import soundfile as sf
import multiprocessing as mp
from multiprocessing import Pool
import random
from dotenv import load_dotenv

load_dotenv()

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")
AUDIOS_FOLDER = os.getenv("AUDIOS_FOLDER")
EMBEDDINGS_GLOBAL_FOLDER = os.getenv("EMBEDDINGS_GLOBAL_FOLDER")
EMBEDDINGS_LOCAL_FOLDER = os.getenv("EMBEDDINGS_LOCAL_FOLDER")

class MusiCNN():
    def __init__(self, ):
        self.model = TensorflowPredictMusiCNN(graphFilename=os.path.join(WEIGHTS_PATH, "msd-musicnn-1.pb"), output="model/dense/BiasAdd")

    def get_audio_embeddings(self, audio_path):
        audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
        embeddings = self.model(audio)

        #take the mean over time and convert to tensor
        embeddings = torch.tensor(embeddings).mean(dim=0, keepdim=True)

        return embeddings  
    
def process_many_files(task):
    dataset, files = task
    model = MusiCNN()

    #make sure the dataset folder exists in the embeddings folder
    dataset_save_path = os.path.join(EMBEDDINGS_FOLDER, dataset)
    os.makedirs(dataset_save_path, exist_ok=True)
    
    #define file extension according to dataset
    if dataset == 'mood_change':
        file_extension = '.wav'
    else:
        file_extension = '.mp3'

    #iterate over the files and extract embeddings
    for i, audio_file in enumerate(files):
        with open('log.txt', 'a') as f:
            f.write(f"Processing row {i+1}/{len(files)}\n")
        
        subdir_save_path = os.path.join(dataset_save_path, os.path.dirname(audio_file))
        os.makedirs(subdir_save_path, exist_ok=True)

        embedding_save_path = os.path.join(subdir_save_path, os.path.basename(audio_file) + '.npy')

        if os.path.exists(embedding_save_path):
            with open('log.txt', 'a') as f:
                f.write(f"Embedding already exists for {audio_file}, skipping.\n")
            continue

        embedding = model.get_audio_embeddings(os.path.join(AUDIOS_FOLDER, dataset, audio_file + file_extension))

        with open('log.txt', 'a') as f:
            f.write(f"Saving embedding for {audio_file} to {embedding_save_path}\n")
        
        np.save(embedding_save_path, embedding.cpu().numpy())
        
        #cleanup
        del embedding
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

    #datasets = ['original']  #to build triplets for REAL DATASET
    datasets = ['mood_change']  #to filter similar triplets in the Controlled Dataset

    with open('log.txt', 'w') as f:
        f.write(f"Starting processing of datasets: {datasets}\n")

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

        with open('log.txt', 'a') as f:
            f.write(f"Dataset {dataset} has {len(audio_files)} audio files, split into {NUM_WORKERS} chunks.\n")

        with Pool(processes=NUM_WORKERS) as pool:
           results = pool.map(process_many_files, tasks)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()