import os
import librosa
import torch
from essentia.standard import TensorflowPredictEffnetDiscogs
from essentia import Pool as EssentiaPool
import gc
from dotenv import load_dotenv

load_dotenv()

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")

class DiscogsEffnet():
    def __init__(self):
        self.model = TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(WEIGHTS_PATH, "discogs_artist_embeddings-effnet-bs64-1.pb"), output="PartitionedCall:1")

    def get_audio_embeddings(self, audio, sr):
        # resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        embeddings = self.model(audio)

        #take the mean over time and convert to tensor
        embeddings = torch.tensor(embeddings).mean(dim=0, keepdim=True)

        return embeddings 