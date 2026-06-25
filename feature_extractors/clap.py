import torch
import numpy as np
import pandas as pd
import librosa as rosa
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from transformers import AutoProcessor, ClapModel

class HuggingFaceClap:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for clap")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")


    def get_audio_tokens(self, audio_path):
        audio_data, _ = rosa.load(audio_path, sr=48000)  # shape (T,)
        audio_data = audio_data.reshape(1, -1)  # (1, T)

        input_text = [""]
        inputs = self.processor(text=input_text, audios=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        #local tokens, of shape [L,D], where L is the number of tokens and D is the dimension of the embeddings
        audio_tokens = outputs.audio_model_output.last_hidden_state  # [1, 1024, 2, 32]
        audio_tokens = audio_tokens.squeeze(0)  # [1024, 2, 32]
        audio_tokens = audio_tokens.permute(1, 2, 0)  # [2, 32, 1024]
        emb_size = audio_tokens.shape[-1]  # 1024
        local_audio_tokens = audio_tokens.reshape(-1, emb_size)  # [64, 1024]

        global_audio_tokens = outputs.audio_embeds

        return global_audio_tokens, local_audio_tokens

    def get_text_tokens(self, input_text):
        #generate fake audio data for testing
        audio_data = np.random.rand(1, 48000)

        input_text = [input_text]

        inputs = self.processor(text=input_text, audio=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        global_text_tokens = outputs.text_embeds

        #local tokens, of shape [L,D], where L is the number of tokens and D is the dimension of the embeddings
        local_text_tokens = outputs.text_model_output.last_hidden_state 

        return global_text_tokens, local_text_tokens

