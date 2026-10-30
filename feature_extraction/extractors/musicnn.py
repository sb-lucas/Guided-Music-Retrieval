import os
import librosa
import torch
from essentia.standard import TensorflowPredictMusiCNN, MonoLoader
from essentia import Pool as EssentiaPool
import gc
from dotenv import load_dotenv

load_dotenv()

WEIGHTS_FOLDER = os.getenv("WEIGHTS_FOLDER")

class MusiCNN():
    has_local_tokens = False        # MusiCNN has only a global embedding

    def __init__(self, ):
        self.model = TensorflowPredictMusiCNN(graphFilename=os.path.join(WEIGHTS_FOLDER, "msd-musicnn-1.pb"), output="model/dense/BiasAdd")

    def get_audio_tokens(self, audio_path):
        audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
        embeddings = torch.tensor(self.model(audio)).mean(dim=0, keepdim=True)

        #global tokens, of shape [1, D], and no local tokens
        return embeddings, None

    def get_audio_embeddings(self, audio, sr):
        # resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        embeddings = self.model(audio)

        #take the mean over time and convert to tensor
        embeddings = torch.tensor(embeddings).mean(dim=0, keepdim=True)

        return embeddings  