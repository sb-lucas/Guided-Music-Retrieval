import os
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import gc
from dotenv import load_dotenv

load_dotenv()

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")

class MERT():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = "m-a-p/MERT-v1-95M"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device).eval()

    def get_audio_embeddings(self, audio, sr):
        # Resample to match MERT's expected sample rate 
        resample_rate = self.processor.sampling_rate
        if sr != resample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=resample_rate)
            
        # The processor expects a 1D numpy array or raw tensor
        audio = audio.squeeze()
        
        inputs = self.processor(audio, sampling_rate=resample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # take a look at the output shape, there are 13 layers of representation
        # each layer performs differently in different downstream tasks, you should choose empirically
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze() # [13 layer, Time steps, 768 feature_dim]

        # for utterance level classification tasks, you can simply reduce the representation in time
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2) # [13, 768]
    
        final_embedding = time_reduced_hidden_states[-1].unsqueeze(0) # [1,768]

        return final_embedding