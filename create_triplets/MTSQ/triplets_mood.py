import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import librosa
import numpy as np
import pandas as pd
import os
import random

JAMENDO_PATH = 'path/to/audios' 
CLASSIFICATIONS_PATH = "path/to/song_labels_multilabel.csv"
GENRE_PATH = "path/to/song_genre.csv"
BINARY_CLASSIFICATIONS_PATH = "path/to/song_labels_binary.csv"

def get_song_tags(instrument_row, theme_row, genre_row):
    filtered_instruments = instrument_row[instrument_row>0.1]
    sorted_instruments = filtered_instruments.sort_values(ascending=False)
    instruments = sorted_instruments.head(4).index.tolist()

    if len(instruments) == 0:
        return None

    genre, subgenre = genre_row['genre'].split('---')

    return genre, subgenre, instruments

def create_new_song(model, output_path, tags, new_mood):
    if os.path.exists(output_path+'.wav'):
        print(f"File {output_path} already exists. Skipping generation.")
        return f"Saved generated song at {output_path}"

    genre, subgenre, instruments = tags
    descriptions = [f"A {new_mood} {genre} {subgenre} song with only {', '.join(instruments)}"]
    wav = model.generate(descriptions)  

    for idx, one_wav in enumerate(wav):
        # Will save under {generated_song_path}.wav, with loudness normalization at -14 db LUFS.
        audio_write(output_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    return f"Saved generated song at {output_path}"

def get_unpresent_moods(row):
    unpresent_moods = []
    
    # We use a threshold to ensure the activation is practically dead.
    THRESHOLD = 0.05 

    #----------------------------- Happy -----------------------------#
    cols = ['happy', 'fun', 'funny', 'positive', 'upbeat', 'summer', 'children']
    if (row[cols] < THRESHOLD).all():
        unpresent_moods.append('happy')

    #----------------------------- Sad -----------------------------#
    cols = ['sad', 'melancholic', 'emotional']
    if (row[cols] < THRESHOLD).all():
        unpresent_moods.append('sad')

    #----------------------------- Energetic -----------------------------#
    cols = ['energetic', 'fast', 'party',  'heavy', 'action', 'sport']
    if (row[cols] < THRESHOLD).all():
        unpresent_moods.append('energetic')

    #----------------------------- Calm -----------------------------#
    cols = ['calm', 'relaxing', 'meditative', 'slow', 'soft', 'nature', 'soundscape']
    if (row[cols] < THRESHOLD).all():
        unpresent_moods.append('calm')

    #----------------------------- Epic -----------------------------#
    cols = ['epic', 'inspiring', 'motivational', 'uplifting', 'hopeful', 'adventure', 'drama', 'dramatic', 'emotional', 'powerful']
    if (row[cols] < THRESHOLD).all():
        unpresent_moods.append('epic')

    #----------------------------- Dark -----------------------------#
    cols = ['dark', 'deep', 'heavy']
    if (row[cols] < THRESHOLD).all():
        unpresent_moods.append('dark')

    #----------------------------- Romantic -----------------------------#
    cols = ['romantic', 'love', 'sexy', 'ballad']
    if (row[cols] < THRESHOLD).all():
        unpresent_moods.append('romantic')

    return unpresent_moods

def process_many_songs(audio_files):
    #load model
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=10.0)

    #load classifications
    df = pd.read_csv(CLASSIFICATIONS_PATH, index_col=0).set_index('song')

    instrument_df = df.drop(columns=[col for col in df.columns if col.startswith('theme_')])
    instrument_df.columns = instrument_df.columns.str.replace('^instrument_', '', regex=True)

    mood_df = df.drop(columns=[col for col in df.columns if col.startswith('instrument_')])
    mood_df.columns = mood_df.columns.str.replace('^theme_', '', regex=True)

    genre_df = pd.read_csv(GENRE_PATH).set_index('song')

    #loop through songs
    triplets = []
    for idx, audio_file in enumerate(audio_files):
        print(f"Processing file {idx+1}/{len(audio_files)}: {audio_file}")
        
        no_ext_audio_file = audio_file.split('.')[0]
        original_song_save_name = os.path.join('original', audio_file)
        
        tags = get_song_tags(instrument_df.loc[no_ext_audio_file], mood_df.loc[no_ext_audio_file], genre_df.loc[no_ext_audio_file])
        unpresent_moods = get_unpresent_moods(mood_df.loc[no_ext_audio_file])

        for new_mood in unpresent_moods:
            target_song_save_name = os.path.join('mood_change', audio_file).split('.')[0] + '_' + new_mood
            save_path = os.path.join(JAMENDO_PATH, target_song_save_name)
            msg = create_new_song(model, save_path, tags, new_mood)
            print(msg)

            new_triplet = {
                'reference_song': original_song_save_name+'.mp3',
                'modification': 'make it ' + new_mood,
                'target_song': target_song_save_name+".wav",
                'dataset':'mood_change'
            }

            complementary_triplet = {
                'reference_song': target_song_save_name+".wav",
                'modification': 'make it not ' + new_mood,
                'target_song': original_song_save_name+'.mp3',
                'dataset':'mood_change'
            }
            
            triplets.append(new_triplet)
            triplets.append(complementary_triplet)

    return triplets

def main():
    #Music Gen can only generate vocaless songs. To better guarantee similarity between the generated and original songs, 
    # we will only use songs without vocals. To make sure a song does not have vocals, we use essentia classifiers.
    df_bin = pd.read_csv(BINARY_CLASSIFICATIONS_PATH, index_col=0)
    df_bin = df_bin[['voice_instrumental','confidence_voice_instrumental']]   
    df_bin = df_bin[df_bin['voice_instrumental'] == 'instrumental']
    df_bin = df_bin[(df_bin['confidence_voice_instrumental'] > 0.6)]
    audio_files = df_bin.index.to_list()
    audio_files = sorted(audio_files)

    print(f"Number of instrumental songs with high confidence: {len(audio_files)}")
    
    triplets = process_many_songs(audio_files)

    # Save triplets to a CSV file
    triplets_df = pd.DataFrame(triplets)
    triplets_df.to_csv('save/path', index=False)

if __name__ == "__main__":
    main()