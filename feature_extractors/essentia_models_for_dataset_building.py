import sys
from read_tsv import read_tsv
import pandas as pd
import multiprocessing as mp
import os
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D, TensorflowPredictVGGish
import os

class EssentiaModels:
    def __init__(self, weights_folder):
        self.weights_folder = weights_folder
        
        # Core embedding model
        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=f"{weights_folder}/discogs-effnet-bs64-1.pb",
            output="PartitionedCall:1"
        )

        #aqui
        # self.embedding_model = TensorflowPredictVGGish(
        #     graphFilename=f"{weights_folder}/audioset-vggish-3.pb",
        #     output="model/vggish/embeddings"
        # )

        #Binary models
        self.binary_models_classes = {
            "danceability": ["danceable", "not danceable"],
            "mood_aggressive": ["aggressive", "not aggressive"],
            "mood_happy": ["happy", "not happy"],
            "mood_party": ["not party", "party"],
            "mood_relaxed": ["not relaxed", "relaxed"],
            "mood_sad": ["sad", "not sad"],
            "mood_acoustic": ["acoustic", "not acoustic"],
            "mood_electronic": ["electronic", "not electronic"],
            "voice_instrumental": ["instrumental", "voice"],
            "gender": ["female", "male"],
            "timbre": ["bright", "dark"],
        }


        self.models = {}

        for name in self.binary_models_classes.keys():
            print(f"Loading model for {name}...")
            self.models[name] = TensorflowPredict2D(
                graphFilename=f"{weights_folder}/{name}-discogs-effnet-1.pb",
                output="model/Softmax"
            )

        #Multilabel models
        #aqui
        self.multilabel_models_classes = {
            "instrument": ['accordion', 'acousticbassguitar', 'acousticguitar', 'bass', 'beat', 'bell', 'bongo', 
                            'brass', 'cello', 'clarinet', 'classicalguitar', 'computer', 'doublebass', 'drummachine', 
                            'drums', 'electricguitar', 'electricpiano', 'flute', 'guitar', 'harmonica', 'harp', 
                            'horn', 'keyboard', 'oboe', 'orchestra', 'organ', 'pad', 'percussion', 'piano', 
                            'pipeorgan', 'rhodes', 'sampler', 'saxophone', 'strings', 'synthesizer', 'trombone', 
                            'trumpet', 'viola', 'violin', 'voice'
            ],
            "moodtheme": ['action', 'adventure', 'advertising', 'background', 'ballad', 'calm', 'children', 
                          'christmas', 'commercial', 'cool', 'corporate', 'dark', 'deep','documentary', 
                          'drama', 'dramatic', 'dream', 'emotional', 'energetic', 'epic', 'fast', 'film', 'fun', 
                          'funny', 'game', 'groovy', 'happy', 'heavy','holiday', 'hopeful', 'inspiring', 'love', 
                          'meditative', 'melancholic', 'melodic', 'motivational', 'movie', 'nature', 'party', 
                          'positive','powerful', 'relaxing', 'retro', 'romantic', 'sad', 'sexy', 'slow', 'soft', 
                          'soundscape', 'space', 'sport', 'summer', 'trailer', 'travel', 'upbeat','uplifting'
            ]
        }
        
        self.models["instrument"] = TensorflowPredict2D(graphFilename=f"{weights_folder}/mtg_jamendo_instrument-discogs-effnet-1.pb")
        self.models["moodtheme"] = TensorflowPredict2D(graphFilename=f"{weights_folder}/mtg_jamendo_moodtheme-discogs-effnet-1.pb")


    def get_embeddings(self, audio):
        audio = MonoLoader(filename=audio, sampleRate=16000, resampleQuality=4)()
        return self.embedding_model(audio)

    def get_binary_label(self, embeddings, model_name):
        model = self.models[model_name]
        predictions = model(embeddings).mean(axis=0)

        max_index = predictions.argmax()
        label = self.binary_models_classes[model_name][max_index]
        confidence = predictions[max_index]

        return label, confidence

    def get_multiclass_labels(self, embeddings, model_name):
        model = self.models[model_name]
        
        with open('log.txt', 'a') as f:
            f.write(f"Getting predictions for {model_name}...\r\n")
            f.write(f"predicitions shape: {model(embeddings).shape}\r\n")
        
        predictions = model(embeddings).mean(axis=0)
        
        with open('log.txt', 'a') as f:
            f.write(f"after mean predicitions shape: {predictions.shape}\r\n")

        indexes = [i for i, score in enumerate(predictions)]
        labels = [self.multilabel_models_classes[model_name][i] for i in indexes]
        confidences = [predictions[i] for i in indexes]
        
        return labels, confidences

#--------------------------------------- Generate labels for each song --------------------------------------------#
def process_subdir(subdir):
    print(f"Processing subdir: {subdir}")
    try:
        with open('log.txt', 'a') as f:
            f.write(f"Entrei no subdir_{subdir}\r\n")

        dataset_folder = "path/to/dataset" 
        songs_folder = os.path.join(dataset_folder, "audios/")

        #initialize models inside the process to avoid GPU memory issues
        essentia_models = EssentiaModels(weights_folder="../weights")

        binary_models = list(essentia_models.binary_models_classes.keys())
        multilabel_models = list(essentia_models.multilabel_models_classes.keys())

        multilabel_labels = [essentia_models.multilabel_models_classes[i] for i in multilabel_models]
        multilabel_labels = [label for sublist in multilabel_labels for label in sublist]

        all_models = binary_models + multilabel_models
        rows_binary = []
        rows_multilabel = []

        # Process each song in the subdirectory
        subdir_path = os.path.join(songs_folder, subdir)
        counter = 0
        for song in os.listdir(subdir_path):
            with open('log.txt', 'a') as f:
                f.write(f"processando song: {song}\r\n")
            song_path = os.path.join(subdir_path, song)
            embeddings = essentia_models.get_embeddings(song_path)
            
            new_row_binary = {"song": subdir + '/' + song.split('.')[0]}
            new_row_multilabel = {"song": subdir + '/' + song.split('.')[0]}

            for model in all_models:
                with open('log.txt', 'a') as f:
                    f.write(f"Processing model {model} for song {song}\r\n")
                if model in essentia_models.binary_models_classes:
                    label, confidence = essentia_models.get_binary_label(embeddings, model)
                    new_row_binary[model] = label
                    new_row_binary['confidence_' + model] = confidence
                else:
                    labels, confidences = essentia_models.get_multiclass_labels(embeddings, model)
                    for label, confidence in zip(labels, confidences):
                        new_row_multilabel[label] = confidence

            rows_binary.append(new_row_binary)
            rows_multilabel.append(new_row_multilabel)
            counter += 1

            with open('log.txt', 'a') as f:
                f.write(f"Processed {counter} songs\r\n")

        return rows_binary, rows_multilabel
    
    except Exception as e:
        print(f"Error processing subdir {subdir}: {e}")
        with open("errors.txt", "a") as f:
            f.write(f"{subdir}: {e}\n")
        return []

if __name__ == "__main__":
    dataset_folder = "path/to/dataset"  # Replace with the actual path to your dataset
    songs_folder = os.path.join(dataset_folder, "audios/")


    mp.set_start_method("spawn", force=True)

    subdirs = [
        d for d in os.listdir(songs_folder)
        if os.path.isdir(os.path.join(songs_folder, d))
    ]

    try:
        with mp.Pool(processes=10) as pool:  # adjust to VRAM
            results_binary, results_multilabel = zip(*pool.map(process_subdir, subdirs))

    except KeyboardInterrupt:
        print("Interrupted, terminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    labels_binary_df = pd.DataFrame([row for sub in results_binary for row in sub])
    labels_multilabel_df = pd.DataFrame([row for sub in results_multilabel for row in sub])

    # # #-------------------------------------------------- Save to CSV --------------------------------------------------#
    labels_binary_df.to_csv(os.path.join(dataset_folder, "song_labels_binary.csv"), index=False)
    labels_multilabel_df.to_csv(os.path.join(dataset_folder, "song_labels_multilabel.csv"), index=False)


   

