import librosa
import numpy as np
import os 
import soundfile as sf
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

SEPARATED_STEMS_FOLDER = 'path/to/separated/stems'
JAMENDO_PATH = 'path/to/audios'

def rms_energy(y):
    return np.sqrt(np.mean(y**2))

def is_instrument_present(stem_name, stem_path, mixture_path):
    if stem_name == 'vocals':
        rel_thresh=0.4
    elif stem_name == 'drums':
        rel_thresh=0.4
    elif stem_name == 'bass':
        rel_thresh=0.6

    stem, _= librosa.load(stem_path, sr=None)
    mixture, _ = librosa.load(mixture_path, sr=None)

    stem_rms = rms_energy(stem)
    mix_rms = rms_energy(mixture)

    if mix_rms == 0:
        return False

    rel_energy = stem_rms / mix_rms

    return rel_energy > rel_thresh

def merge_stems(stems, out_path, skipped_stem: str):

    filtered_stems = {k: v for k, v in stems.items() if k != skipped_stem}

    loaded_stems = {}
    for k, v in filtered_stems.items():
        loaded_stems[k], sr = librosa.load(v, sr=None)

    # ensure same length
    min_len = min(len(stem) for stem in loaded_stems.values())
    for k in loaded_stems:
        loaded_stems[k] = loaded_stems[k][:min_len]

    # sum stems
    mix = sum(loaded_stems.values())

    # prevent clipping
    max_val = np.max(np.abs(mix))
    if max_val > 1:
        mix = mix / max_val

    sf.write(out_path, mix, sr)

def get_stems_paths(song_name):

    folder = song_name.split('/')[-1].split('.mp3')[0]
    folder_path = os.path.join(SEPARATED_STEMS_FOLDER, folder)

    stems = {
        'vocals': os.path.join(folder_path, 'vocals.wav'),
        'drums': os.path.join(folder_path, 'drums.wav'),
        'bass': os.path.join(folder_path, 'bass.wav'),
        'other': os.path.join(folder_path, 'other.wav')
    }

    return stems

def create_triplets_from_song(song_name):
    reference_song_root_name = song_name.split('.mp3')[0]
    reference_song_subfolder = reference_song_root_name.split('/')[0]
    reference_song_file_name = reference_song_root_name.split('/')[1]
    reference_song_path = os.path.join(os.path.join(JAMENDO_PATH, 'original'), song_name)
    reference_song_save_name = os.path.join('original', song_name)

    triplets = []

    stems = get_stems_paths(song_name)

    for stem, stem_path in stems.items():
        if stem == 'other':
            continue
        if is_instrument_present(stem, stem_path, reference_song_path):
            #we only remove the instrument if it is present to avoid redundant triplets
            #this means we only save valid modificated audios

            target_song_subfolder_path = os.path.join(JAMENDO_PATH, 'remove_instrument', reference_song_subfolder)
            os.makedirs(target_song_subfolder_path, exist_ok=True)
            target_song_save_name = os.path.join(
                                                'remove_instrument', 
                                                reference_song_subfolder, 
                                                reference_song_file_name + f"_no_{stem}" + ".mp3"
                                                )

            save_path = os.path.join(JAMENDO_PATH, target_song_save_name)

            merge_stems(stems, save_path, stem)

            new_triplet = {
                'reference_song': reference_song_save_name,
                'modification': 'remove ' + stem,
                'target_song': target_song_save_name,
                'dataset':'instrumentation_remove'
            }
            complementary_triplet = {
                'reference_song': target_song_save_name,
                'modification': 'add ' + stem,
                'target_song': reference_song_save_name,
                'dataset':'instrumentation_remove'
            }
            triplets.append(new_triplet)
            triplets.append(complementary_triplet)

    return triplets
            
def main():
    audio_files = []
    original_subfolder = os.path.join(JAMENDO_PATH, 'original')
    for i, subfolder in enumerate(os.listdir(original_subfolder)):
        subfolder_path = os.path.join(original_subfolder, subfolder)
        for j, file in enumerate(os.listdir(subfolder_path)):
            if file.endswith('.mp3'):
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
                executor.submit(create_triplets_from_song, song_name) 
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
    df.to_csv("save/path", index=False)

    print(f"Saved {len(df)} triplets to triplets_instrumentation_remove.csv")

if __name__ == "__main__":
    main()