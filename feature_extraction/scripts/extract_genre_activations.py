import pandas as pd
import numpy as np
import os
import sys
import ast
import gc
import sys, os
from read_tsv import read_tsv
import pandas as pd
import os
from essentia import Pool
from essentia.standard import MonoLoader, TensorflowPredictMAEST, TensorflowPredict


#EXTRACT AUDIO TOKENS
def main():
    #folders
    root_path = "path/to/dataset"
    
    embedding_model = TensorflowPredictMAEST(graphFilename="path/to/essentia/weights/discogs-maest-10s-pw-2.pb", output="PartitionedCall/Identity_12")
    model= TensorflowPredict(graphFilename="path/to/essentia/weights/genre_discogs400-discogs-maest-10s-pw-1.pb", inputs=["embeddings"], outputs=["PartitionedCall/Identity_1", "PartitionedCall/Identity"])
    
    genre_activations_folder = os.path.join(root_path, "extracted_features/genre_activations")
    genre_logits_folder = os.path.join(root_path, "extracted_features/genre_logits")

    os.makedirs(genre_activations_folder, exist_ok=True)
    os.makedirs(genre_logits_folder, exist_ok=True)

    audios_folder = os.path.join(root_path, "audios")
    count = sum(
        1
        for root, _, files in os.walk(audios_folder)
        for f in files
        if f.lower().endswith(".mp3")
    )
    counter = 0

    with open(os.path.join(root_path, "log.txt"), 'w') as f:
        f.write(f"Total number of audio files to process: {count}\n")
        f.write("Starting processing...\n")

    #main loop
    for subdir in os.listdir(audios_folder):
        audio_subdir_path = os.path.join(audios_folder, subdir)
        
        genre_subdir_path =  os.path.join(genre_activations_folder, subdir)
        genre_logits_subdir_path = os.path.join(genre_logits_folder, subdir)

        os.makedirs(genre_subdir_path, exist_ok=True)
        os.makedirs(genre_logits_subdir_path, exist_ok=True)

        for audio in os.listdir(audio_subdir_path):
            with open(os.path.join(root_path, "log.txt"), 'a') as f:
                f.write(f"Processing {counter+1}/{count}, audio = {subdir}/{audio}\n")

            audio_path = os.path.join(audio_subdir_path, audio)
            base = audio.split('.')[0]

            genre_activations_path = os.path.join(genre_subdir_path, base + ".npy")
            genre_logits_path = os.path.join(genre_logits_subdir_path, base + ".npy")

            if os.path.exists(genre_activations_path) and os.path.exists(genre_logits_path):
                with open(os.path.join(root_path, "log.txt"), 'a') as f:
                    f.write(f"Activations and logits for {base} already exist, skipping...\n")
            else:
                audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
                embeddings = embedding_model(audio)
                pool = Pool()
                pool.set("embeddings", embeddings)
                activations = model(pool)["PartitionedCall/Identity_1"]
                logits = model(pool)["PartitionedCall/Identity"]

                logits = np.squeeze(logits)
                activations = np.squeeze(activations)

                np.save(genre_activations_path, activations)
                np.save(genre_logits_path, logits)
                
                del activations, logits
                gc.collect()

            counter += 1

if __name__ == "__main__":
    print("começo")
    main()

