#use music_audioset_epoch_15_esc_90.14.pt
import torch
import numpy as np
import librosa
import laion_clap
from laion_clap.training.data import get_audio_features
from laion_clap.hook import int16_to_float32, float32_to_int16
import torch.nn.functional as F
from dotenv import load_dotenv
import os 

load_dotenv()

WEIGHTS_FOLDER = os.getenv("WEIGHTS_FOLDER")

class LAION_CLAP:
    sample_rate = 48000
    clip_samples = 480000      # 10 s, clip_samples of the audio_cfg of HTSAT-base
    local_stride = 32          # fine_grained_embedding has 1024 steps, but only 32 distinct ones, each repeated 32 times

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', tmodel='roberta', device=str(self.device))
        self.module.load_ckpt(os.path.join(WEIGHTS_FOLDER, 'music_audioset_epoch_15_esc_90.14.pt'))
        self.module.eval()
        self.model = self.module.model      # the network, with encode_audio, audio_projection, text_branch and text_projection

    def load_audio(self, audio_path):
        audio_data, _ = librosa.load(audio_path, sr=self.sample_rate)
        audio_data = int16_to_float32(float32_to_int16(audio_data))     # the quantization that the library applies
        return torch.from_numpy(audio_data).float()                     # [T]

    @torch.no_grad()
    def get_audio_tokens(self, audio_path):
        waveform = self.load_audio(audio_path)

        # same call as CLAP_Module.get_audio_embedding_from_data (non fusion model)
        features = get_audio_features({}, waveform, self.clip_samples, data_truncating='rand_trunc',
                                      data_filling='repeatpad', audio_cfg=self.module.model_cfg['audio_cfg'])
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in features.items()}
        out = self.model.encode_audio(batch, device=self.device)

        #global tokens, of shape [1, D]
        global_audio_tokens = F.normalize(self.model.audio_projection(out['embedding']), dim=-1)    # [1, 512]

        #local tokens, of shape [L, D]
        local_audio_tokens = out['fine_grained_embedding'][0, ::self.local_stride]                  # [32, 1024]

        return global_audio_tokens, local_audio_tokens

    @torch.no_grad()
    def get_text_tokens(self, text):
        tokens = self.module.tokenizer([text])          # padded to 77
        attention_mask = tokens['attention_mask'].to(self.device)
        hidden = self.model.text_branch(input_ids=tokens['input_ids'].to(self.device), attention_mask=attention_mask)

        #global tokens, of shape [1, D]
        global_text_tokens = F.normalize(self.model.text_projection(hidden['pooler_output']), dim=-1)   # [1, 512]

        #local tokens, of shape [L, D], without the padding
        local_text_tokens = hidden['last_hidden_state'][0, :attention_mask.sum()]                       # [L, 768]

        return global_text_tokens, local_text_tokens
