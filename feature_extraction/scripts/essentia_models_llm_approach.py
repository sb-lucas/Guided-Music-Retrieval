import sys
from xml.parsers.expat import model
from read_tsv import read_tsv
import pandas as pd
import multiprocessing as mp
import os
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import os
import random
import ast

DATASET_FOLDER = "path/to/dataset" 
SONGS_FOLDER = os.path.join(DATASET_FOLDER, "audios/")

class EssentiaModels:
    def __init__(self, weights_folder):
        self.weights_folder = weights_folder
        
        # Core embedding model
        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=f"{weights_folder}/discogs_label_embeddings-effnet-bs64-1.pb",
            output="PartitionedCall:1"
        )

        self.models = {}

        #Multilabel models
        #aqui
        self.multilabel_models_classes = {
            "instrument": ["accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", 
                           "brass", "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", 
                           "drums", "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", 
                           "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", 
                           "rhodes", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", 
                           "violin", "voice"], 
            
            "moodtheme": ["action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas", 
                          "commercial", "cool", "corporate", "dark", "deep", "documentary", "drama", "dramatic", "dream", 
                          "emotional", "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy", "happy", 
                          "heavy", "holiday", "hopeful", "inspiring", "love", "meditative", "melancholic", "melodic", 
                          "motivational", "movie", "nature", "party", "positive", "powerful", "relaxing", "retro", 
                          "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space", "sport", "summer", "trailer", 
                          "travel", "upbeat", "uplifting"],

            "genre": ["60s", "70s", "80s", "90s", "acidjazz", "alternative", "alternativerock", "ambient", "atmospheric", 
                      "blues", "bluesrock", "bossanova", "breakbeat", "celtic", "chanson", "chillout", "choir", 
                      "classical", "classicrock", "club", "contemporary", "country", "dance", "darkambient", "darkwave", 
                      "deephouse", "disco", "downtempo", "drumnbass", "dub", "dubstep", "easylistening", "edm", 
                      "electronic", "electronica", "electropop", "ethno", "eurodance", "experimental", "folk", "funk", 
                      "fusion", "groove", "grunge", "hard", "hardrock", "hiphop", "house", "idm", "improvisation", 
                      "indie", "industrial", "instrumentalpop", "instrumentalrock", "jazz", "jazzfusion", "latin", 
                      "lounge", "medieval", "metal", "minimal", "newage", "newwave", "orchestral", "pop", "popfolk", 
                      "poprock", "postrock", "progressive", "psychedelic", "punkrock", "rap", "reggae", "rnb", "rock", 
                      "rocknroll", "singersongwriter", "soul", "soundtrack", "swing", "symphonic", "synthpop", "techno", 
                      "trance", "triphop", "world", "worldfusion"]
        }
        
        self.multilabel_models_classes['genre'] = [f"genre_{label}" for label in self.multilabel_models_classes['genre']]
        self.multilabel_models_classes['moodtheme'] = [f"moodtheme_{label}" for label in self.multilabel_models_classes['moodtheme']]
        self.multilabel_models_classes['instrument'] = [f"instrument_{label}" for label in self.multilabel_models_classes['instrument']]

        
        self.models["instrument"] = TensorflowPredict2D(graphFilename=f"{weights_folder}/mtg_jamendo_instrument-discogs_label_embeddings-effnet-1.pb")
        self.models["moodtheme"] = TensorflowPredict2D(graphFilename=f"{weights_folder}/mtg_jamendo_moodtheme-discogs_label_embeddings-effnet-1.pb")
        self.models["genre"] = TensorflowPredict2D(graphFilename=f"{weights_folder}/mtg_jamendo_genre-discogs_label_embeddings-effnet-1.pb")

    def get_embeddings(self, audio):
        audio = MonoLoader(filename=audio, sampleRate=16000, resampleQuality=4)()
        return self.embedding_model(audio)

    def get_multiclass_labels(self, embeddings, model_name):
        model = self.models[model_name]
        
        predictions = model(embeddings).mean(axis=0)
        
        indexes = [i for i, score in enumerate(predictions)]
        labels = [self.multilabel_models_classes[model_name][i] for i in indexes]
        confidences = [predictions[i] for i in indexes]
        
        return labels, confidences

#--------------------------------------- Generate labels for each song --------------------------------------------#
def process_many_files(songs):

    #initialize models inside the process to avoid GPU memory issues
    essentia_models = EssentiaModels(weights_folder="path/to/essentia/weights")

    multilabel_models = list(essentia_models.multilabel_models_classes.keys())

    multilabel_labels = [essentia_models.multilabel_models_classes[i] for i in multilabel_models]
    multilabel_labels = [label for sublist in multilabel_labels for label in sublist]

    rows_multilabel = []

    counter = 0
    for song in songs:
        song_path = os.path.join(SONGS_FOLDER, song)
        embeddings = essentia_models.get_embeddings(song_path)
        
        new_row_multilabel = {"song": song}

        for model in multilabel_models:
            labels, confidences = essentia_models.get_multiclass_labels(embeddings, model)
            for label, confidence in zip(labels, confidences):
                new_row_multilabel[label] = confidence

        rows_multilabel.append(new_row_multilabel)
        counter += 1

        with open('log.txt', 'a') as f:
            f.write(f"Processed {counter}/{len(songs)}: {song}\n")

    return rows_multilabel

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
    splits_real = pd.read_csv('path/to/MTIQ/splits.csv')
    splits_controlled = pd.read_csv('path/to/MTSQ/splits.csv')

    real_test_songs = splits_real[splits_real['split'] == 'test']['song'].to_list()
    real_test_songs = [f"original/{song}.mp3" for song in real_test_songs]
    controlled_test_songs = splits_controlled[splits_controlled['split'] == 'test']['song'].to_list()

    all_unique_songs = list(set(real_test_songs + controlled_test_songs))

    NUM_WORKERS = 16

    chunks = split_list(all_unique_songs, NUM_WORKERS)

    with open(os.path.join("log.txt"), 'w') as f:
        f.write(f"Starting processing of {len(all_unique_songs)} unique songs, split into {NUM_WORKERS} chunks.\n")

    try:
        with mp.Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(process_many_files, chunks)

    except KeyboardInterrupt:
        print("Interrupted, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    # cada elemento de `results` é o retorno de um chunk
    flat_results = [item for sublist in results for item in sublist]
    parsed = [ast.literal_eval(x) if isinstance(x, str) else x for x in flat_results]
    labels_multilabel_df = pd.DataFrame(parsed)
    labels_multilabel_df.to_csv("save/path", index=False)
   
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

    
