import os
import librosa
import torch
from essentia.standard import TensorflowPredictMAEST
from essentia import Pool as EssentiaPool
import gc
from dotenv import load_dotenv

load_dotenv()

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")


class MAEST():
    def __init__(self, ):
        self.model = TensorflowPredictMAEST(graphFilename=os.path.join(WEIGHTS_PATH, "discogs-maest-5s-pw-2.pb"), output="PartitionedCall/Identity_7")

    def get_audio_embeddings(self, audio, sr):
        # resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Pad to minimum of 5s (5.1 for a safety margin)
        target_samples = int(5.1 * target_sr)
        audio = librosa.util.fix_length(audio, size=target_samples)

        #get embeddings and extract the relevant layer
        embeddings = self.model(audio)
        embeddings = torch.tensor(embeddings)
        embeddings = embeddings[0,0,0,:].unsqueeze(0) #get the relevant layer and add batch dimension

        return embeddings  