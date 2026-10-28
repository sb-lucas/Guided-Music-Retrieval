import os
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

class Demucs():
    def __init__(self, model_name="htdemucs", device=None):
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device} for demucs")

        #the model is loaded once and reused for every song
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.sources = self.model.sources  # ['drums', 'bass', 'other', 'vocals']
        self.samplerate = self.model.samplerate
        self.channels = self.model.audio_channels

    def separate(self, audio_path, shifts=1, overlap=0.25):
        #returns a dict {stem name: tensor of shape [channels, T]} at the model's sample rate
        wav = AudioFile(audio_path).read(streams=0, samplerate=self.samplerate, channels=self.channels)

        #same input normalization as the demucs command line tool
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(self.model, wav[None], device=self.device, shifts=shifts,
                                  split=True, overlap=overlap, progress=False)[0]
        sources = sources * ref.std() + ref.mean()

        return {name: source.cpu() for name, source in zip(self.sources, sources)}

    def separate_to_folder(self, audio_path, save_folder, shifts=1, overlap=0.25):
        #saves the stems as <save_folder>/<model name>/<track name>/<stem>.wav, same layout as the demucs command line tool
        stems = self.separate(audio_path, shifts=shifts, overlap=overlap)

        track_name = os.path.splitext(os.path.basename(audio_path))[0]
        track_folder = os.path.join(save_folder, self.model_name, track_name)
        os.makedirs(track_folder, exist_ok=True)

        stem_paths = {}
        for name, stem in stems.items():
            stem_paths[name] = os.path.join(track_folder, f"{name}.wav")
            save_audio(stem, stem_paths[name], samplerate=self.samplerate)

        return stem_paths
