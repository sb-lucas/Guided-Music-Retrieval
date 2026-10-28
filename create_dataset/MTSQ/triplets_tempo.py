import soundfile as sf
import pyrubberband as pyrb
import numpy as np
import os
import random
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

ROOT_FOLDER = os.getenv("ROOT_FOLDER")
JAMENDO_PATH = os.getenv("AUDIOS_FOLDER")
TRIPLETS_PATH = os.path.join(ROOT_FOLDER, "data", "MTSQ", "triplets_tempo.csv")

def slow_down_audio(audio_file, rate):
    input_path = os.path.join(JAMENDO_PATH, 'original', audio_file)
    output_path = os.path.join(JAMENDO_PATH, 'slowed', audio_file)

    input_song, sr = sf.read(input_path)

    #Play back at speed defined by rate (e.g., 0.5 for half speed)
    slowed_song = pyrb.time_stretch(input_song, sr, rate)

    #ensure same length by cutting the slowed down audio to the original length
    #using the middle portion
    original_length = len(input_song)
    slowed_length = len(slowed_song)
    if slowed_length > original_length:
        start_idx = (slowed_length - original_length) // 2
        slowed_song = slowed_song[start_idx:start_idx + original_length]

    #save the slowed down audio
    subdir = audio_file.split('/')[0]  
    os.makedirs(os.path.join(JAMENDO_PATH, 'slowed', subdir), exist_ok=True)
    sf.write(output_path, slowed_song, sr)

    return f"Audio slowed down by a factor of {rate} and saved to {output_path}"

def create_tempo_triplets(audio_file):

    triplets = []

    if os.path.exists(os.path.join(JAMENDO_PATH, 'slowed', audio_file)):
        print(f"File {audio_file} already exists in slowed folder. Skipping generation.")
    else:
        rate = random.uniform(0.5, 0.9) # Randomly choose a rate between 0.5x and 0.9x for slowing down
        slow_down_audio(audio_file, rate)

    original_song_save_name = os.path.join('original', audio_file)
    target_song_save_name = os.path.join('slowed', audio_file)

    new_triplet = {
        'reference_song': original_song_save_name,
        'modification': 'make it slower',
        'target_song': target_song_save_name,
        'dataset':'tempo'
    }
    complementary_triplet = {
        'reference_song': target_song_save_name,
        'modification': 'make it faster',
        'target_song': original_song_save_name,
        'dataset':'tempo'
    }
    
    triplets.append(new_triplet)
    triplets.append(complementary_triplet)

    return triplets

def main():
    audio_files = []
    original_subfolder = os.path.join(JAMENDO_PATH, 'original')
    for i, subfolder in enumerate(sorted(os.listdir(original_subfolder))):
        subfolder_path = os.path.join(original_subfolder, subfolder)
        for j, file in enumerate(sorted(os.listdir(subfolder_path))):
            song_name = os.path.join(subfolder, file)
            audio_files.append(song_name)

    tam_audio = len(audio_files)

    max_workers = 16

    all_triplets = []

    print("Starting audio extraction...")
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(create_tempo_triplets, song_name) 
                for song_name in audio_files
            ]

            # Merge results as they complete
            for i, future in enumerate(as_completed(futures_audio)):
                try:
                    result = future.result()
                    if result:  # avoid None / empty
                        all_triplets.extend(result)
                    print(f"{i+1}/{tam_audio} processed")
                except Exception as e:
                    print(f"FAILED on audio {i + 1}. Error: {e}")

    except KeyboardInterrupt:
        print("Process interrupted by user. Shutting down.")
        # Cancel pending tasks + stop accepting new ones
        executor.shutdown(wait=False, cancel_futures=True)

        # Force kill remaining child processes
        for p in multiprocessing.active_children():
            p.terminate()

        print("All workers terminated.")

    # --- SAVE CSV ---
    df = pd.DataFrame(all_triplets)
    df.to_csv(TRIPLETS_PATH, index=False)

    print(f"Saved {len(df)} triplets to triplets_tempo.csv")

if __name__ == "__main__":
    main()
