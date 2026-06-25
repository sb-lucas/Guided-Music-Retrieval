import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from transformers import AutoProcessor, ClapModel
import csv
import ast
import torch.nn.functional as F

from models.limn import LIMN
from models.one_layer import MLP
from models.baldrati import Baldrati
from models.tg_cir import TG_CIR
from models.llm import LLM_Based

import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../feature_extractors')))
from clap import HuggingFaceClap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../create_triplets')))
from real.modification_augmentation import modifications, modifications_reverse


class ModelEvaluationBase:
    def __init__(self, model_weights_path, model_architecture, save_path, dataset):
        self.local_audio_folder = 'path/to/clap_local'
        self.global_audio_folder = 'path/to/clap_global'
        self.audios_folder = os.path.join("path/to/audios")

        self.model_architecture =  model_architecture
        self.model_weights_path = model_weights_path
        self.model_weights_complete_path = os.path.join(f"runs/{model_weights_path}", "model_best.pth.tar")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

        self.model_init()

        #---------------------------------- create the matrix of embeddings of targets to use in knn-------------------------------------#
        # Pre-loading embeddings into dicts for faster access during evaluation
        #pre loading all audio embeddings
        self.audio_cache = {}
        if os.path.exists(f"cache/audio_cache_{dataset}_test.pt"):
            print("Loading pre-computed audio cache...")
            self.audio_cache = torch.load(f"cache/audio_cache_{dataset}_test.pt", map_location="cpu")
        else:
            for i, song in enumerate(self.all_songs):
                print(f"Pre-loading audio embeddings: {i+1}/{len(self.all_songs)}. song = {song}: ", end='\r')
                song_id = song.split('.')[0]
                local_path = os.path.join(self.local_audio_folder, f"{song_id}.npy")
                global_path = os.path.join(self.global_audio_folder, f"{song_id}.npy")

                local_ = torch.from_numpy(np.load(local_path)).float().squeeze()
                global_ = torch.from_numpy(np.load(global_path)).float().squeeze()

                # normalize once
                global_ = F.normalize(global_, dim=-1)
                local_ = F.normalize(local_, dim=-1)

                self.audio_cache[song_id] = (global_, local_)
            
            #save cache
            torch.save(self.audio_cache, f"cache/audio_cache_{dataset}_test.pt")

        #pre loading all text embeddings
        self.clap_model = HuggingFaceClap()
        self.text_cache = {}
        unique_texts = self.triplets_df['modification_text'].unique()
        if os.path.exists(f"cache/text_cache_{dataset}_test.pt"):
            print("Loading pre-computed text cache...")
            self.text_cache = torch.load(f"cache/text_cache_{dataset}_test.pt", map_location="cpu")
        else:
            for text in unique_texts:
                g, l = self.clap_model.get_text_tokens(text)
                g = g.detach().cpu()
                l = l.detach().cpu()
                
                l = F.normalize(l.float().squeeze(), dim=-1)
                g = F.normalize(g.float().squeeze(), dim=-1)

                self.text_cache[text] = (g, l)
            #save cache
            torch.save(self.text_cache, f"cache/text_cache_{dataset}_test.pt")

        torch.cuda.empty_cache()  # optional but safe

        #----------------------------------------------knn---------------------------------------------------------#
        if model_architecture == "limn":
            self.build_limn_embeddings_matrix(self.all_songs)
        elif model_architecture == "tg_cir":
            self.build_tg_cir_embeddings_matrix(self.all_songs)
        else:
            self.build_embeddings_matrix(self.all_songs)

        self.k_values = [1, 2, 5, 10, 20, 50, 100]
        max_k = max(self.k_values)+1
        self.knn = NearestNeighbors(n_neighbors=max_k, algorithm='brute', metric='cosine').fit(self.embeddings_matrix)
        
    #-------------------------------------------------------Building Target Embeddings Matrix---------------------------------------------------------#
    def build_embeddings_matrix(self, all_songs):
        #cases when the model does not encode the target, and uses CLAP directly
        self.embeddings_matrix = []
        self.song_to_position = {}
        self.position_to_song = {}

        print(f"Total unique songs: {len(all_songs)}")

        for song in all_songs:
            song_id = song.split('.')[0]

            global_emb = self.audio_cache[song_id][0]  

            self.song_to_position[song_id] = len(self.embeddings_matrix)
            self.position_to_song[len(self.embeddings_matrix)] = song_id

            self.embeddings_matrix.append(global_emb)

        print(f"shape of embeddings matrix: {np.array(self.embeddings_matrix).shape}")

    def build_tg_cir_embeddings_matrix(self, all_songs):
        self.embeddings_matrix = []
        self.song_to_position = {}
        self.position_to_song = {}

        print(f"Total unique songs: {len(all_songs)}")

        for song in all_songs:
            song_id = song.split('.')[0]
            
            global_emb, local_emb = self.audio_cache[song_id]
            global_emb = global_emb.unsqueeze(0).to(self.device)
            local_emb = local_emb.unsqueeze(0).to(self.device)

            with torch.no_grad():
                E_t = self.model(None, None, global_emb, None, None, local_emb)
                mean_pooled_E_t = torch.mean(E_t, dim=1)
                target_embedding = mean_pooled_E_t.detach().cpu().numpy().squeeze()

            self.song_to_position[song_id] = len(self.embeddings_matrix)
            self.position_to_song[len(self.embeddings_matrix)] = song_id

            self.embeddings_matrix.append(target_embedding)

        print(f"shape of embeddings matrix: {np.array(self.embeddings_matrix).shape}")

    def build_limn_embeddings_matrix(self, all_songs):
        self.embeddings_matrix = []
        self.song_to_position = {}
        self.position_to_song = {}

        #dummies of shape torch.Size([B, 512]) torch.Size([B, L, 768])
        f_dummy = torch.zeros((1, 512)).to(self.device)
        e_dummy = torch.zeros((1, 768)).unsqueeze(0).to(self.device)

        for song in all_songs:
            song_id = song.split('.')[0]

            ft, et = self.audio_cache[song_id]
            ft = ft.unsqueeze(0).to(self.device)
            et = et.unsqueeze(0).to(self.device)

            self.song_to_position[song_id] = len(self.embeddings_matrix)
            self.position_to_song[len(self.embeddings_matrix)] = song_id

            with torch.no_grad():
                # def forward(self, fr, fm, ft, er, em, et, eq_len, et_len):
                et_len = torch.tensor([et.shape[1]]).unsqueeze(0)  
                eq_len = torch.tensor([2]).unsqueeze(0)  

                #[DEBUG] ft shape: torch.Size([1, 512]), et shape: torch.Size([1, 64, 768]), f_dummy shape: torch.Size([1, 512]), e_dummy shape: torch.Size([1, 1, 768]), eq_len shape: torch.Size([1, 1]), et_len shape: torch.Size([1, 1])

                ___, __, _, a_hat_t = self.model(f_dummy, f_dummy, ft, e_dummy, e_dummy, et, eq_len, et_len)

                target_emb = a_hat_t.detach().cpu().numpy().squeeze()

            self.embeddings_matrix.append(target_emb)

        print(f"shape of embeddings matrix: {np.array(self.embeddings_matrix).shape}")

    #---------------------------------------------------------Setting Up Each Model ------------------------------------------------#
    def model_init(self):
        if self.model_architecture == "one_layer":
            embedding_dim = 512
            self.model = MLP(input_dim=embedding_dim*2, output_dim=embedding_dim).to(self.device)
            self.model.load_state_dict(torch.load(self.model_weights_complete_path, map_location=torch.device('cpu'))['state_dict'])
            self.model.eval()
            self.model_fusion = self.STD_fusion

        elif self.model_architecture == "baseline":
            self.model = None
            self.alpha = self.model_weights_path[0] #song weight
            self.beta = self.model_weights_path[1] #text weight
            self.model_fusion = self.baseline_fusion

        elif self.model_architecture == "limn":
            self.model = LIMN().to(self.device)
            self.model.load_state_dict(torch.load(self.model_weights_complete_path, map_location=torch.device('cpu'))['state_dict'])
            self.model.eval()
            self.model_fusion = self.limn_fusion

        elif self.model_architecture == 'baldrati':
            self.model = Baldrati(clap_feature_dim=512, projection_dim=256, hidden_dim=128).to(self.device)
            self.model.load_state_dict(torch.load(self.model_weights_complete_path, map_location=torch.device('cpu'))['state_dict'])
            self.model.eval()
            self.model_fusion = self.STD_fusion
        
        elif self.model_architecture == 'tg_cir':
            self.model = TG_CIR().to(self.device)
            self.model.load_state_dict(torch.load(self.model_weights_complete_path, map_location=torch.device('cpu'))['state_dict'])
            self.model.eval()
            self.model_fusion = self.tg_cir_fusion
        
        elif self.model_architecture == 'llm':
            self.model = None
            self.model_fusion = self.LLM_fusion
            model_name = "Qwen/Qwen3-8B"
            self.model = LLM_Based(model_name)
    
    #----------------------------------------------Defining Fusions for Each Model---------------------------------------------------------#
    def LLM_fusion(self, song_name, modification_text):
        target_desc = self.model.get_target_description(song_name, modification_text)

        g, l = self.clap_model.get_text_tokens(target_desc)
        g = g.detach().cpu()
        g = F.normalize(g.float().squeeze(), dim=-1)

        return g.numpy()

    def baseline_fusion(self, fr, fm, er, em):
        fr = fr.detach().cpu().numpy().squeeze()
        fm = fm.detach().cpu().numpy().squeeze()

        sum_embedding = self.alpha*fr + self.beta*fm

        return sum_embedding

    def STD_fusion(self, fr, fm, er, em):

        composed_embedding = self.model(fr, fm)

        return composed_embedding.detach().cpu().numpy()

    def limn_fusion(self, fr, fm, er, em):
        with torch.no_grad():
            eq_len = torch.tensor([er.shape[1] + em.shape[1]]).unsqueeze(0)  
            et_len = torch.tensor([1]).unsqueeze(0)  

            f_dummy = torch.zeros((1, 512)).to(self.device)
            e_dummy = torch.zeros((1, 768)).unsqueeze(0).to(self.device)

            ___, a_hat_q, __, _ = self.model(fr, fm, f_dummy, er, em, e_dummy, eq_len, et_len)
            composed_embedding = a_hat_q.detach().cpu().numpy().squeeze()
        
        return composed_embedding

    def tg_cir_fusion(self, fr, fm, er, em):
        with torch.no_grad():
            E_c_hat = self.model(fr, fm, None, er, em, None)
            mean_pooled_E_c_hat = torch.mean(E_c_hat, dim=1)
            composed_embedding = mean_pooled_E_c_hat.detach().cpu().numpy().squeeze()
        
        return composed_embedding
    
    #---------------------------------------------------------Getting Composed Embedding---------------------------------------------------------#
    def get_composed_embedding(self, row):
        # Get local and global reference song embeddings -> pre-calculated
        reference_song_id = row['reference_song'].split('.')[0]

        ref_global, ref_local = self.audio_cache[reference_song_id]

        #local and global modification text embeddings
        modification_text = row['modification_text']
        mod_global, mod_local = self.text_cache[modification_text]

        #correct the format
        ref_global = ref_global.to(self.device).unsqueeze(0)
        ref_local = ref_local.to(self.device).unsqueeze(0)
        mod_global = mod_global.to(self.device).unsqueeze(0)
        mod_local = mod_local.to(self.device).unsqueeze(0)

        # Get composed embedding using the selected model
        if self.model_architecture == 'llm':
            composed_embedding = self.model_fusion(row['reference_song'], modification_text)
        else:
            composed_embedding = self.model_fusion(ref_global, mod_global, ref_local, mod_local)

        return composed_embedding
    
    
    @staticmethod
    def save_results_csv(csv_path, model_name, weights_path, k_values, raks, rak_types, metric='R@'):
        # converte para path absoluto (mostra onde o arquivo será criado)
        csv_path = os.path.abspath(csv_path)

        # garante que a pasta existe (se quiser outra pasta, ajuste aqui)
        dirpath = os.path.dirname(csv_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        for recall_at_k_dict, rak_type in zip(raks, rak_types):
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                
                # header só se o arquivo não existir ou estiver vazio
                write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
                if write_header:
                    header = ['type', "model", "weights_path"] + [f"{metric}{k}" for k in k_values]
                    writer.writerow(header)

                # monta a linha na mesma ordem de k_values (evita mismatch de order)
                row = [rak_type, model_name, weights_path]
                for k in k_values:
                    val = recall_at_k_dict.get(k, None)
                    if val is None:
                        row.append("")   # ou row.append("nan")
                    else:
                        row.append(f"{float(val):.4f}")
                writer.writerow(row)
        print(f"Results saved to {csv_path}")


class ModelEvaluationMTSQ(ModelEvaluationBase):
    def __init__(self, model_weights_path, model_architecture, save_path):
        self.triplets_df = pd.read_csv('path/to/mtsq_test_triplets.csv')

        splits_df = pd.read_csv('path/to/splits.csv')
        test_split = splits_df[splits_df['split'] == 'test'].set_index('song')

        self.all_songs = test_split.index.tolist()

        super().__init__(model_weights_path, model_architecture, save_path, 'MTSQ')
    
    def evaluate(self):
        RaK_by_type = {}

        for key in self.triplets_df['modification'].unique().tolist():
            RaK_by_type[key] = {}  # initialize first
            RaK_by_type[key]['file'] = {k: 0 for k in self.k_values} 

        RaK_by_type['global'] = {}
        RaK_by_type['global']['file'] = {k: 0 for k in self.k_values} 

        #----------------------------------------------evaluate---------------------------------------------------------#    
        #for each song, i get their recall at k
        for i, (index, row) in enumerate(self.triplets_df.iterrows()):
            if i % 10 == 0:
                print(f"{i}/{len(self.triplets_df)}", end='\r')

            #get composed embedding
            reference_song_id = row['reference_song'].split('.')[0]
            composed_embedding = self.get_composed_embedding(row)

            #get ground truth
            ground_truth = row['target_song'].split('.')[0]

            #get similar songs to the fusioned embedding
            distances, indices = self.knn.kneighbors(composed_embedding.reshape(1, -1))
            filtered_indices = [i for i in indices[0] if i != self.song_to_position[reference_song_id]] # Exclude the reference song itself
            similar_songs = [self.position_to_song[position] for position in filtered_indices]

            #ground truth tag
            modification = row['modification']

            #for each k, i get their recall at k
            for k in self.k_values:
                #FILE RECALL @ K
                #if the ground truth in predicted in the top k predictions, i return 1, else 0
                if ground_truth in similar_songs[:k]:
                    recall = 1
                else:
                    recall = 0
                
                RaK_by_type[modification]['file'][k] += recall
                RaK_by_type['global']['file'][k] += recall

        for modification in RaK_by_type.keys():
            print(f"Recall@K for modification '{modification}':")

            if modification == 'global':
                n_triplets = len(self.triplets_df)
            else:
                n_triplets = len(self.triplets_df[self.triplets_df['modification'] == modification])

            #normalizando RaK
            RaK_by_type[modification]['file'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['file'].items()}
            
            rak_types = ['file']
            raks = [RaK_by_type[modification][rak_type] for rak_type in rak_types]

            #salvando resultados no csv
            csv_path = os.path.join(self.save_path, f"results_{modification}.csv")
            self.save_results_csv(csv_path, self.model_architecture, self.model_weights_path, self.k_values, raks, rak_types)

class ModelEvaluationMTIQ(ModelEvaluationBase):
    def __init__(self, model_weights_path, model_architecture, save_path):
        self.triplets_df = pd.read_csv('path/to/triplets_mtiq_test.csv')

        similarity_df = pd.read_csv('path/to/similarity_mtiq_test.csv')
        self.similarity_df = similarity_df.set_index('reference_song')  # Set 'reference_song' as the index for easy lookup 
        self.similarity_df['similar_songs'] = self.similarity_df['similar_songs'].apply(ast.literal_eval) 

        #merge tags_df with splits_df to get only the songs in the test set
        splits_df = pd.read_csv('path/to/splits.csv')
        test_split = splits_df[splits_df['split'] == 'test']

        tags_df = pd.read_csv('path/to/labels.csv')
        tags_df = tags_df.merge(test_split[['song']], on='song', how='inner')
        tags_df['song'] = 'original/' + tags_df['song'] + '.mp3' #so the idx are in the 'save name'
        self.tags_df = tags_df.set_index('song')  # Set 'song' as the index for easy lookup

        self.all_songs = self.tags_df.index.tolist()

        super().__init__(model_weights_path, model_architecture, save_path, 'mtiq')
    
    def evaluate(self):
        RaK_by_type = {}
        for key in self.triplets_df['modification'].unique().tolist():
            RaK_by_type[key] = {'file': {k: 0 for k in self.k_values}, 
                                'tag': {k: 0 for k in self.k_values}, 
                                'similarity': {k: 0 for k in self.k_values},
                                'both': {k: 0 for k in self.k_values}}
        
        RaK_by_type['global'] = {'file': {k: 0 for k in self.k_values}, 
                                'tag': {k: 0 for k in self.k_values}, 
                                'similarity': {k: 0 for k in self.k_values},
                                'both': {k: 0 for k in self.k_values}}

        #----------------------------------------------evaluate---------------------------------------------------------#    
        #for each song, i get their recall at k
        for i, (index, row) in enumerate(self.triplets_df.iterrows()):
            if i % 10 == 0:
                print(f"{i}/{len(self.triplets_df)}", end='\r')

            #get composed embedding
            reference_song_id = row['reference_song'].split('.')[0]
            composed_embedding = self.get_composed_embedding(row)

            #get ground truth
            ground_truth = row['target_song'].split('.')[0]

            #get similar songs to the fusioned embedding
            distances, indices = self.knn.kneighbors(composed_embedding.reshape(1, -1))
            filtered_indices = [i for i in indices[0] if i != self.song_to_position[reference_song_id]] # Exclude the reference song itself
            similar_songs = [self.position_to_song[position] for position in filtered_indices]

            #get similar songs to the reference song
            similar_songs_to_ref = self.similarity_df.loc[reference_song_id + '.mp3']['similar_songs']
            similar_songs_to_ref = [song.split('.')[0] for song in similar_songs_to_ref]

            #ground truth tag
            modification = row['modification']
            ground_truth_tag = modifications_reverse[modification]

            #for each k, i get their recall at k
            for k in self.k_values:
                #FILE RECALL @ K
                #if the ground truth in predicted in the top k predictions, i return 1, else 0
                if ground_truth in similar_songs[:k]:
                    recall = 1
                else:
                    recall = 0
                
                RaK_by_type[modification]['file'][k] += recall
                RaK_by_type['global']['file'][k] += recall

                #SIMILARITY RECALL @ K
                #check if at least one of the top k predictions is a similar song to the reference
                if any(song in similar_songs[:k] for song in similar_songs_to_ref):
                    recall = 1
                else:
                    recall = 0

                RaK_by_type[modification]['similarity'][k] += recall
                RaK_by_type['global']['similarity'][k] += recall
            
                #TAG RECALL @ K
                #if the ground truth tag appears in the top k predictions, I return 1, else 0
                #get tags of top k predictions
                top_k_tags = set()
                for song in similar_songs[:k]:
                    tags = self.tags_df.loc[song+'.mp3']
                    top_k_tags.update(tag for tag in tags if tag != 'uncertain')
                if ground_truth_tag in top_k_tags:
                    recall = 1
                else:
                    recall = 0
                RaK_by_type[modification]['tag'][k] += recall
                RaK_by_type['global']['tag'][k] += recall

                #BOTH RECALL @ K
                #if at least one song is similar, and the tag is correct, i return 1, else 0
                
                # get matching similar songs within top-k
                matched_songs = [song for song in similar_songs_to_ref if song in similar_songs[:k]]

                if matched_songs:  # at least one match exists
                    matched_tags = set()
                    for song in matched_songs:
                        tags = self.tags_df.loc[song+'.mp3']
                        matched_tags.update(tag for tag in tags if tag != 'uncertain')
                    if ground_truth_tag in matched_tags: #at least one of the matched songs has the correct tag
                        recall = 1
                    else:
                        recall = 0
                else:
                    recall = 0

                RaK_by_type[modification]['both'][k] += recall
                RaK_by_type['global']['both'][k] += recall


        for modification in RaK_by_type.keys():
            print(f"Recall@K for modification '{modification}':")

            if modification == 'global':
                n_triplets = len(self.triplets_df)
            else:
                n_triplets = len(self.triplets_df[self.triplets_df['modification'] == modification])

            #normalizando RaK
            RaK_by_type[modification]['file'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['file'].items()}
            RaK_by_type[modification]['tag'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['tag'].items()}
            RaK_by_type[modification]['similarity'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['similarity'].items()}
            RaK_by_type[modification]['both'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['both'].items()}
            rak_types = ['file', 'tag', 'similarity', 'both']
            raks = [RaK_by_type[modification][rak_type] for rak_type in rak_types]

            #salvando resultados no csv
            csv_path = os.path.join(self.save_path, f"results_{modification}.csv")
            self.save_results_csv(csv_path, self.model_architecture, self.model_weights_path, self.k_values, raks, rak_types)








































    # def inference(self, reference_song, modification_text, target_song):
#     # Get local and global reference song embeddings
#         reference_song_id = reference_song.split('.')[0]
#         local_audio_path = os.path.join(self.local_audio_folder, reference_song_id+".npy")
#         global_audio_path = os.path.join(self.global_audio_folder, reference_song_id+".npy")
        
#         fr = torch.from_numpy(np.load(global_audio_path)).to(self.device)
#         er = torch.from_numpy(np.load(local_audio_path)).unsqueeze(0).to(self.device)

#         #local and global modification text embeddings
#         fm, em = self.clap.get_text_tokens(modification_text)

#         # Get composed embedding using the selected model
#         composed_embedding = self.model_fusion(fr, fm, er, em)

#         #get similar songs to the fusioned embedding
#         distances, indices = self.knn.kneighbors(composed_embedding.reshape(1, -1))
#         filtered_indices = [i for i in indices[0] if i != self.song_to_position[reference_song_id]] # Exclude the reference song itself
#         similar_songs = [self.position_to_song[position] for position in filtered_indices]

#         #get similar songs to the reference song
#         similar_songs_to_ref_str = self.similarity_df[self.similarity_df['reference_song'] == reference_song_id+'.mp3']['similar_songs'].values[0]
#         similar_songs_to_ref = ast.literal_eval(similar_songs_to_ref_str)

#         # info to return
#         k=3

#         retrieved_songs_info = []
#         for song in similar_songs[:k]:
#             tags = self.tags_df.loc[song]
#             tags = [tag for tag in tags if tag != 'uncertain']
        
#             is_similar = song in similar_songs_to_ref
#             retrieved_songs_info.append({
#                 "song": song,
#                 "tags": tags,
#                 "is_similar": True if song in similar_songs_to_ref else False
#             })

#         return retrieved_songs_info