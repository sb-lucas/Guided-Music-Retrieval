import sys
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
        
        predictions = model(embeddings).mean(axis=0)
        
        indexes = [i for i, score in enumerate(predictions)]
        labels = [self.multilabel_models_classes[model_name][i] for i in indexes]
        confidences = [predictions[i] for i in indexes]
        
        return labels, confidences