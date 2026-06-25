#filtrar similaridade igual fiz no Real dataset(acho q 90%)
import os
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from modification_augmentation import augment_modification


MUSICNN_EMBEDDINGS_FOLDER = "path/to/musicnn/embeddings"  # Update this path to where your MusicNN embeddings are stored

#---------------------------------load datasets---------------------------------#
triplets_mood_df = pd.read_csv('path/to/triplets_mood.csv')
triplets_velocity_df = pd.read_csv('path/to/triplets_velocity.csv')
triplets_instrumentation_add_df = pd.read_csv('path/to/triplets_instrumentation_add.csv')
triplets_instrumentation_remove_df = pd.read_csv('path/to/triplets_instrumentation_remove.csv')

#------------------------------------------------Filter Mood Triplets by Similarity------------------------------------------------#
threshold = 90
target_songs = triplets_mood_df['target_song'].to_list()
referece_songs = triplets_mood_df['reference_song'].to_list()
all_unique_songs = set(target_songs + referece_songs)

print(f"Total unique songs in mood triplets: {len(all_unique_songs)}")

embeddings_cache = {}
for song in all_unique_songs:
    print(f"Processing ({len(embeddings_cache)}/{len(all_unique_songs)})", end='\r')
    if song.endswith('.mp3'):
        embedding_sim_path = os.path.join(MUSICNN_EMBEDDINGS_FOLDER, song.replace('.mp3', '.npy'))
    elif song.endswith('.wav'):
        embedding_sim_path = os.path.join(MUSICNN_EMBEDDINGS_FOLDER, song.replace('.wav', '.npy'))
    
    embedding_sim = np.load(embedding_sim_path) # shape [D,]
    embedding_sim = normalize(embedding_sim.reshape(1, -1), axis=1).squeeze()
    embeddings_cache[song] = embedding_sim

filtered_triplets = []
for index, triplet in triplets_mood_df.iterrows():
    reference_song = triplet['reference_song']
    target_song = triplet['target_song']

    ref_emb = embeddings_cache[reference_song]
    target_emb = embeddings_cache[target_song]

    #get cosine sim
    cos_sim = cosine_similarity(ref_emb.reshape(1, -1), target_emb.reshape(1, -1))[0][0]

    #convert similarity to percentage (0 to 100)
    sim_percent = ((cos_sim + 1) / 2) * 100

    if sim_percent >= threshold:
        filtered_triplets.append(triplet)

filtered_mood_df = pd.DataFrame(filtered_triplets)
print(f"Original mood triplets: {len(triplets_mood_df)}")
print(f"Filtered mood triplets: {len(filtered_mood_df)}")

#------------------------------------------------Modification Text Augmentation------------------------------------------------#
filtered_mood_df['modification_text'] = filtered_mood_df['modification'].apply(augment_modification)
triplets_velocity_df['modification_text'] = triplets_velocity_df['modification'].apply(augment_modification)
triplets_instrumentation_add_df['modification_text'] = triplets_instrumentation_add_df['modification'].apply(augment_modification)
triplets_instrumentation_remove_df['modification_text'] = triplets_instrumentation_remove_df['modification'].apply(augment_modification)

#---------------------------------------------Merge all Datasets--------------------------------------------------#
merged_df = pd.concat([filtered_mood_df, triplets_velocity_df, triplets_instrumentation_add_df, triplets_instrumentation_remove_df], ignore_index=True)
merged_df.to_csv('path/to/final_triplets.csv', index=False)

#--------------------------------------------Split Train val and test------------------------------------------#
splits_df = pd.read_csv('path/to/splits.csv')
train_songs = splits_df[splits_df['split'] == 'train']['song'].to_list()
test_songs = splits_df[splits_df['split'] == 'test']['song'].to_list()
val_songs = splits_df[splits_df['split'] == 'val']['song'].to_list()

splits_controlled = []

target_songs = merged_df['target_song'].to_list()
referece_songs = merged_df['reference_song'].to_list()
all_unique_songs = set(target_songs + referece_songs)

for index, song in enumerate(all_unique_songs):
    print(f"processing {index} / {len(all_unique_songs)}", end='\r')

    root_name = song.split('.')[0]  # remove file extension
    root_name = '/'.join(root_name.split('/')[1:]) #remove dataset
    root_name = root_name.split('_')[0] #remove modification

    if root_name in test_songs:
        splits_controlled.append({
            'song': song,
            'split': 'test'
        })
    elif root_name in val_songs:
        splits_controlled.append({
            'song': song,
            'split': 'val'
        })
    else:
        splits_controlled.append({
            'song': song,
            'split': 'train'
        })

splits_controlled_df = pd.DataFrame(splits_controlled)
splits_controlled_df.to_csv('path/to/splits.csv', index=False)

train_songs = splits_controlled_df[splits_controlled_df['split'] == 'train']['song'].to_list()
train_triplets_df = merged_df[merged_df['reference_song'].isin(train_songs) & merged_df['target_song'].isin(train_songs)]
train_triplets_df.to_csv('path/to/train_triplets.csv', index=False)
                         
val_songs = splits_controlled_df[splits_controlled_df['split'] == 'val']['song'].to_list()
val_triplets_df = merged_df[merged_df['reference_song'].isin(val_songs) & merged_df['target_song'].isin(val_songs)]
val_triplets_df.to_csv('path/to/val_triplets.csv', index=False)

test_songs = splits_controlled_df[splits_controlled_df['split'] == 'test']['song'].to_list()
test_triplets_df = merged_df[merged_df['reference_song'].isin(test_songs) & merged_df['target_song'].isin(test_songs)]
test_triplets_df.to_csv('path/to/test_triplets.csv', index=False)