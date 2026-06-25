import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import shutil
from xml.parsers.expat import model
import demucs.separate
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import shutil


TEMP_SEPARATED_PATH = 'path/to/temp/separated'
SEPARATED_PATH = "path/to/separated/htdemucs"
JAMENDO_PATH = 'path/to/audios'
CLASSIFICATIONS_PATH = "path/to/song_labels_multilabel.csv"

def separate_audio(file_path):
    try:
        demucs.separate.main([
                "-o", TEMP_SEPARATED_PATH,
                file_path
            ])
        return f"Successfully separated audio for {file_path}"
    except Exception as e:
        print(f"Error occurred while separating audio for {file_path}: {e}")

def rms_energy(y):
    return np.sqrt(np.mean(y**2))

def is_instrument_present(stem_path, mixture_path, rel_thresh=0.9):

    stem, _= librosa.load(stem_path, sr=None)
    mixture, _ = librosa.load(mixture_path, sr=None)

    stem_rms = rms_energy(stem)
    mix_rms = rms_energy(mixture)

    if mix_rms == 0:
        return False

    rel_energy = stem_rms / mix_rms

    return rel_energy > rel_thresh

def merge_stems(reference_song_path, new_stem_path, out_path):


    ref_song, sr = librosa.load(reference_song_path, sr=None)
    new_stem, _ = librosa.load(new_stem_path, sr=None)

    # ensure same length
    min_len = min(len(ref_song), len(new_stem))
    ref_song = ref_song[:min_len]
    new_stem = new_stem[:min_len]

    # sum stems
    mix = ref_song + new_stem

    # prevent clipping
    max_val = np.max(np.abs(mix))
    if max_val > 1:
        mix = mix / max_val

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, mix, sr)

def create_new_song(model, instrument, reference_melody_path, reference_song_save_name):
    song_name = reference_song_save_name.split('.')[0].split('/')[-1]
    subfolder = song_name[-2:]
    
    save_song_name = os.path.join('add_instrument', subfolder, song_name + f'_{instrument}.mp3')
    save_path = os.path.join(JAMENDO_PATH, save_song_name)
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping generation.")
        return save_song_name


    melody, sr = torchaudio.load(reference_melody_path)

    #--generate melody--#
    wav = model.generate_with_chroma(
        descriptions=[f"isolated {instrument} melody"], 
        melody_wavs=melody.expand(1, -1, -1), 
        melody_sample_rate=sr
    )

    generated_song_path = f'{song_name}_{instrument}_melody'

    #saving
    for idx, one_wav in enumerate(wav):
        # Will save under {generated_song_path}.wav, with loudness normalization at -14 db LUFS.
        audio_write(generated_song_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    #--clean and validate melody--#
    
    #since the output of the model might contain more instruments
    #than the one i asked, i extract only what i asked for 
    #what i want is in the 'other' stem
    separate_audio(generated_song_path + '.wav')
    stems_path = os.path.join(TEMP_SEPARATED_PATH, 'htdemucs', generated_song_path)
    other_stem_path = os.path.join(stems_path, 'other.wav')
    
    #--mix and save final song--#
    merge_stems(other_stem_path, os.path.join(JAMENDO_PATH, reference_song_save_name), save_path)

    #delete generated files to save space
    os.remove(generated_song_path + '.wav')
    shutil.rmtree(stems_path)

    return save_song_name

def get_unpresent_instruments(row):
    
    instruments = []
    
    #to make sure an instrument is not on the song,
    #we make sure no similar sounding instrument is present in the song

    #-----------------------------Saxophone-----------------------------#
    cols = ['trombone', 'trumpet', 'horn', 'brass', 'clarinet', 'flute', 'oboe', 'saxophone']
    if (row[cols] < 0.05).all():
        instruments.append('saxophone')

    #-----------------------------Flute-----------------------------#
    cols = ['clarinet', 'flute', 'oboe', 'saxophone']
    if (row[cols] < 0.05).all():
        instruments.append('flute')
    
    #-----------------------------Piano-----------------------------#
    cols = ['electricpiano','keyboard','rhodes','organ','piano','pipeorgan','accordion']
    if (row[cols] < 0.20).all():
        instruments.append('piano')
    
    #-----------------------------Guitar-----------------------------#
    cols = ['acousticguitar', 'classicalguitar', 'electricguitar', 'guitar', 'harp']
    if (row[cols] < 0.15).all():
        instruments.append('acoustic guitar')

    #-----------------------------Violin-----------------------------#
    cols = ['cello', 'strings', 'violin', 'viola', 'orchestra']
    if (row[cols] < 0.05).all():
        instruments.append('violin')

    return instruments

def process_many_files(audio_files):
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=10.0)

    df = pd.read_csv(CLASSIFICATIONS_PATH, index_col=0).set_index('song')
    df.drop(columns=[col for col in df.columns if col.startswith('theme_')], inplace=True)
    df.columns = df.columns.str.replace('^instrument_', '', regex=True)

    triplets = []
    for i, audio_file in enumerate(audio_files):
        print(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")

        original_song_save_name = os.path.join('original', audio_file)

        stem_folder_name = audio_file.split('/')[1].split('.')[0]
        stem_folder_path = os.path.join(SEPARATED_PATH, stem_folder_name)
        other_stem_path = os.path.join(stem_folder_path, 'other.wav')
        
        unpresent_instruments = get_unpresent_instruments(df.loc[audio_file.split('.')[0]])

        for instrument in unpresent_instruments:
            target_song_save_name = create_new_song(model, instrument, other_stem_path, original_song_save_name)

            new_triplet = {
                'reference_song': original_song_save_name,
                'modification': 'add ' + instrument,
                'target_song': target_song_save_name,
                'dataset':'instrumentation_add'
            }
            complementary_triplet = {
                'reference_song': target_song_save_name,
                'modification': 'remove ' + instrument,
                'target_song': original_song_save_name,
                'dataset':'instrumentation_add'
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
                
    df = pd.read_csv(CLASSIFICATIONS_PATH, index_col=0).set_index('song')
    df.drop(columns=[col for col in df.columns if col.startswith('theme_')], inplace=True)
    df.columns = df.columns.str.replace('^instrument_', '', regex=True)

    triplets = process_many_files(audio_files)

    # Save triplets to a CSV file
    triplets_df = pd.DataFrame(triplets)
    triplets_df.to_csv('save/path', index=False)
    

if __name__ == "__main__":
    main()