import torch
import numpy as np
import librosa as rosa
from transformers import ClapModel, ClapProcessor

class HF_CLAP:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for clap")
        self.model = ClapModel.from_pretrained("laion/larger_clap_music").to(self.device)
        self.model.eval()
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

    def get_audio_tokens(self, audio_path):
        audio_data, _ = rosa.load(audio_path, sr=48000)  # shape (T,)

        input_text = [""]
        inputs = self.processor(text=input_text, audio=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        #local tokens, of shape [L,D], where L is the number of tokens and D is the dimension of the embeddings
        audio_tokens = outputs.audio_model_output.last_hidden_state   # [1, C, freq_bins, time]
        audio_tokens = audio_tokens.squeeze(0)                        # [C, freq_bins, time]
        audio_tokens = audio_tokens.permute(2, 1, 0)                  # [time, freq_bins, C]
        local_audio_tokens = audio_tokens.mean(dim=1)                 # [time, C]

        #global tokens, of shape [1, D], where D is the dimension of the embeddings
        global_audio_tokens = outputs.audio_embeds #[1, 512]

        return global_audio_tokens, local_audio_tokens

    def get_text_tokens(self, input_text):
        #generate fake audio data for testing
        audio_data = np.random.rand(48000)

        input_text = [input_text]

        inputs = self.processor(text=input_text, audio=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        #global tokens, of shape [1, D], where D is the dimension of the embeddings
        global_text_tokens = outputs.text_embeds #[1, 512]

        #local tokens, of shape [L,D], where L is the number of tokens and D is the dimension of the embeddings
        local_text_tokens = outputs.text_model_output.last_hidden_state.squeeze(0)  # [L, D]

        return global_text_tokens, local_text_tokens