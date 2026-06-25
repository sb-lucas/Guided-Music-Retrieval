import pandas as pd
import os
import librosa
from transformers import AutoProcessor, ClapModel
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch.nn.functional as F
from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN, TensorflowPredictMAEST
from pathlib import Path
import essentia.standard as es
import numpy as np
from essentia import Pool as EssentiaPool
import gc
import soundfile as sf
import multiprocessing as mp
from multiprocessing import Pool

WEIGHTS_PATH = 'path/to/essentia/weights' 

#--------------------------------------DEFINING ALL MODELS---------------------------------------#
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

class CLAP():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    def get_audio_embeddings(self, audio, sr):
        #resample to 48kHz if needed
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)

        audio_data = audio.reshape(1, -1)  # (1, T)

        input_text = [""]
        inputs = self.processor(text=input_text, audios=audio_data, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)

        outputs = self.model(**inputs)

        return outputs.audio_embeds

#---------------------------OPENL3-----------------------------#
class MelSpectrogramOpenL3:
    def __init__(self, hop_time):
        self.hop_time = hop_time

        self.sr = 48000
        self.n_mels = 128
        self.frame_size = 2048
        self.hop_size = 242
        self.a_min = 1e-10
        self.d_range = 80
        self.db_ref = 1.0

        self.patch_samples = int(1 * self.sr)
        self.hop_samples = int(self.hop_time * self.sr)

        self.w = es.Windowing(
            size=self.frame_size,
            normalized=False,
        )
        self.s = es.Spectrum(size=self.frame_size)
        self.mb = es.MelBands(
            highFrequencyBound=self.sr / 2,
            inputSize=self.frame_size // 2 + 1,
            log=False,
            lowFrequencyBound=0,
            normalize="unit_tri",
            numberBands=self.n_mels,
            sampleRate=self.sr,
            type="magnitude",
            warpingFormula="slaneyMel",
            weighting="linear",
        )

    def compute(self, audio_file):
        audio = es.MonoLoader(filename=audio_file, sampleRate=self.sr)()

        batch = []
        for audio_chunk in es.FrameGenerator(
            audio, frameSize=self.patch_samples, hopSize=self.hop_samples
        ):
            melbands = np.array(
                [
                    self.mb(self.s(self.w(frame)))
                    for frame in es.FrameGenerator(
                        audio_chunk,
                        frameSize=self.frame_size,
                        hopSize=self.hop_size,
                        validFrameThresholdRatio=0.5,
                    )
                ]
            )

            melbands = 10.0 * np.log10(np.maximum(self.a_min, melbands))
            melbands -= 10.0 * np.log10(np.maximum(self.a_min, self.db_ref))
            melbands = np.maximum(melbands, melbands.max() - self.d_range)
            melbands -= np.max(melbands)

            batch.append(melbands.copy())
        return np.vstack(batch)


class EmbeddingsOpenL3:
    def __init__(self, graph_path, hop_time=1, batch_size=60, melbands=128):
        self.hop_time = hop_time
        self.batch_size = batch_size

        self.graph_path = Path(graph_path)

        self.x_size = 199
        self.y_size = melbands
        self.squeeze = False

        self.permutation = [0, 3, 2, 1]

        self.input_layer = "melspectrogram"
        self.output_layer = "embeddings"

        self.mel_extractor = MelSpectrogramOpenL3(hop_time=self.hop_time)

        self.model = es.TensorflowPredict(
            graphFilename=str(self.graph_path),
            inputs=[self.input_layer],
            outputs=[self.output_layer],
            squeeze=self.squeeze,
        )

    def compute(self, audio_file):
        mel_spectrogram = self.mel_extractor.compute(audio_file)
        # in OpenL3 the hop size is computed in the feature extraction level

        hop_size_samples = self.x_size

        batch = self.__melspectrogram_to_batch(mel_spectrogram, hop_size_samples)

        pool = EssentiaPool()
        embeddings = []
        nbatches = int(np.ceil(batch.shape[0] / self.batch_size))
        for i in range(nbatches):
            start = i * self.batch_size
            end = min(batch.shape[0], (i + 1) * self.batch_size)
            pool.set(self.input_layer, batch[start:end])
            out_pool = self.model(pool)
            embeddings.append(out_pool[self.output_layer].squeeze())

        return np.vstack(embeddings)

    def __melspectrogram_to_batch(self, melspectrogram, hop_time):
        npatches = int(np.ceil((melspectrogram.shape[0] - self.x_size) / hop_time) + 1)
        batch = np.zeros([npatches, self.x_size, self.y_size], dtype="float32")
        for i in range(npatches):
            last_frame = min(i * hop_time + self.x_size, melspectrogram.shape[0])
            first_frame = i * hop_time
            data_size = last_frame - first_frame

            # the last patch may be empty, remove it and exit the loop
            if data_size <= 0:
                batch = np.delete(batch, i, axis=0)
                break
            else:
                batch[i, :data_size] = melspectrogram[first_frame:last_frame]

        batch = np.expand_dims(batch, 1)
        batch = es.TensorTranspose(permutation=self.permutation)(batch)
        return batch

class OpenL3():
    def __init__(self):
        # load the model
        self.model = EmbeddingsOpenL3(os.path.join(WEIGHTS_PATH, "openl3-music-mel128-emb512-3.pb"))

    def get_audio_embeddings(self, audio, sr):
        #save temporary audio file
        sf.write("audio.wav", audio, sr)

        embeddings = self.model.compute("audio.wav")

        #delete temporary audio file
        os.remove("audio.wav")

        #take the mean over time and convert to tensor
        embeddings = torch.tensor(embeddings).mean(dim=0, keepdim=True)

        return embeddings

#---------------------------OPENL3-----------------------------#

class VGGISH():
    def __init__(self):
        self.model = TensorflowPredictVGGish(graphFilename=os.path.join(WEIGHTS_PATH,"audioset-vggish-3.pb"), output="model/vggish/embeddings")

    def get_audio_embeddings(self, audio, sr):
        # resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        embeddings = self.model(audio)

        #take the mean over time and convert to tensor
        embeddings = torch.tensor(embeddings).mean(dim=0, keepdim=True)

        return embeddings  
    
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

class MusiCNN():
    def __init__(self, ):
        self.model = TensorflowPredictMusiCNN(graphFilename=os.path.join(WEIGHTS_PATH, "msd-musicnn-1.pb"), output="model/dense/BiasAdd")

    def get_audio_embeddings(self, audio, sr):
        # resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        embeddings = self.model(audio)

        #take the mean over time and convert to tensor
        embeddings = torch.tensor(embeddings).mean(dim=0, keepdim=True)

        return embeddings  

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
    
#--------------------------------------EXPERIMENT CODE---------------------------------------#

def load_audio(song_id, start_samples, clip_length_samples, sr):
    file_path = f'path/to/dimsim/audios/{song_id}.mp3'
    audio, _ = librosa.load(file_path, sr=sr, offset=start_samples/sr, duration=clip_length_samples/sr)
    return audio

def solve_triplet(model, sr, anchor, song1, song2):
    #extract embeddings with model
    anchor_emb = model.get_audio_embeddings(anchor, sr)
    song1_emb = model.get_audio_embeddings(song1, sr)
    song2_emb = model.get_audio_embeddings(song2, sr)

    #ensure normalized embeddings
    anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
    song1_emb = F.normalize(song1_emb, p=2, dim=1)
    song2_emb = F.normalize(song2_emb, p=2, dim=1)

    #calculate cosine similarity
    sim_song1 = F.cosine_similarity(anchor_emb, song1_emb)
    sim_song2 = F.cosine_similarity(anchor_emb, song2_emb)

    #return 1 if song1 is more similar to the anchor, else return 2
    if sim_song1 > sim_song2:
        return 1
    else:        
        return 2
    


def evaluate_model(ModelClass):
    df = pd.read_csv('path/to/dimsim/metadata/clean-dim-sim.csv')

    model_name = ModelClass.__name__
    with open('log.txt', 'a') as f:
        f.write(f"Initializing model: {model_name}\n")

    model = ModelClass()
    counter = 0

    for index, row in df.iterrows():
        with open('log.txt', 'a') as f:
            f.write(f"Processing row {index+1}/{len(df)} for model {model_name}\n")

        sr = row['sampling_rate']

        anchor_audio = load_audio(row['anchor_id'], row['anchor_start_samples'], row['clip_lengths_samples'], sr)
        song1_audio = load_audio(row['song1_id'], row['song1_start_samples'], row['clip_lengths_samples'], sr)
        song2_audio = load_audio(row['song2_id'], row['song2_start_samples'], row['clip_lengths_samples'], sr)

        model_sim = solve_triplet(model, sr, anchor_audio, song1_audio, song2_audio)

        gt_sim = 1 if row['song1_vote'] == 1 else 2

        if model_sim == gt_sim:
            counter += 1

    acc = counter / len(df)

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (model_name, acc)

def main():
    model_classes = [VGGISH, MAEST, CLAP, MusiCNN, DiscogsEffnet, MERT, OpenL3]

    NUM_WORKERS = 7

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(evaluate_model, model_classes)

    accs_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
    accs_df.to_csv('save/path', index=False)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()