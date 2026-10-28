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

from dotenv import load_dotenv
from feature_extraction.extractors.vggish import VGGISH
from feature_extraction.extractors.maest import MAEST
from feature_extraction.extractors.clap import CLAP
from feature_extraction.extractors.discogs_effnet import DiscogsEffnet
from feature_extraction.extractors.musicnn import MusiCNN
from feature_extraction.extractors.mert import MERT
from feature_extraction.extractors.openl3 import MelSpectrogramOpenL3
    
#--------------------------------------EXPERIMENT CODE---------------------------------------#

def load_audio(song_id, start_samples, clip_length_samples, sr):
    file_path = f'/home/ldsbrandao/exp/diego/diego/final/dimsim/clean_dimsim_audios/{song_id}.mp3'
    audio, _ = librosa.load(file_path, sr=sr, offset=start_samples/sr, duration=clip_length_samples/sr)
    return audio

def solve_triplet(model, sr, anchor, song1, song2):
    #extract embeddings with model
    anchor_emb = model.get_audio_embeddings(anchor, sr)
    song1_emb = model.get_audio_embeddings(song1, sr)
    song2_emb = model.get_audio_embeddings(song2, sr)

    #ensure normalized embeddings
    anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
    song1_emb = F.normalize(song1_emb, p=2, dim=1)
    song2_emb = F.normalize(song2_emb, p=2, dim=1)

    #calculate cosine similarity
    sim_song1 = F.cosine_similarity(anchor_emb, song1_emb)
    sim_song2 = F.cosine_similarity(anchor_emb, song2_emb)

    #return 1 if song1 is more similar to the anchor, else return 2
    if sim_song1 > sim_song2:
        return 1
    else:        
        return 2
    

def evaluate_model(ModelClass):
    df = pd.read_csv('/home/ldsbrandao/exp/diego/diego/final/dimsim/metadata/clean-dim-sim.csv')

    model_name = ModelClass.__name__
    with open('log.txt', 'a') as f:
        f.write(f"Initializing model: {model_name}\n")

    model = ModelClass()
    counter = 0

    for index, row in df.iterrows():
        with open('log.txt', 'a') as f:
            f.write(f"Processing row {index+1}/{len(df)} for model {model_name}\n")

        sr = row['sampling_rate']

        anchor_audio = load_audio(row['anchor_id'], row['anchor_start_samples'], row['clip_lengths_samples'], sr)
        song1_audio = load_audio(row['song1_id'], row['song1_start_samples'], row['clip_lengths_samples'], sr)
        song2_audio = load_audio(row['song2_id'], row['song2_start_samples'], row['clip_lengths_samples'], sr)

        model_sim = solve_triplet(model, sr, anchor_audio, song1_audio, song2_audio)

        gt_sim = 1 if row['song1_vote'] == 1 else 2

        if model_sim == gt_sim:
            counter += 1

    acc = counter / len(df)

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (model_name, acc)

def main():
    #model_classes = [VGGISH, MAEST, CLAP, MusiCNN, DiscogsEffnet, MERT, OpenL3]

    #NUM_WORKERS = 7

    # with Pool(processes=NUM_WORKERS) as pool:
    #     results = pool.map(evaluate_model, model_classes)

    # accs_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    # accs_df.to_csv('/home/ldsbrandao/exp/diego/diego/final/dimsim/accuracies.csv', index=False)
    
    results = evaluate_model(CLAP)
    print(results)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()