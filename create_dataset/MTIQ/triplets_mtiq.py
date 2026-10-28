import os
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import gc
import re
import ast
import sys
from modification_augmentation import get_modification_from_tag, augment_modification

DATASET_FOLDER = '/path/to/dataset'
MUSICNN_EMBEDDINGS_FOLDER = os.path.join(DATASET_FOLDER, 'extracted_features/musicnn/')
TRIPLETS_FOLDER = '/path/to/triplets'



def select_tags(df_binary):
    # Define confidence thresholds for each tag
    confidences ={
        'danceability': 0.98,
        'mood_aggressive': 0.97,
        'mood_happy': 0.97,
        'mood_party': 0.98,
        'mood_relaxed': 0.95,
        'mood_sad': 0.94,
        'mood_acoustic': 0.95,
        'mood_electronic': 0.98,
        'voice_instrumental': 0.99,
    }

    df_binary = df_binary.set_index('song')
    df_binary.index = df_binary.index.astype(str).str.strip()

    #convert the dataset of tag/confidence to a dataset of tags only, 
    #with 'uncertain' for those that are below the confidence threshold
    rows_binary = []
    binary_columns = df_binary.columns.tolist()
    binary_tags = [col for col in binary_columns if not col.startswith('confidence_')]
    #ignore gender and timbre because the models are very bad at this and only a few triplets would be created based on these
    binary_tags.remove('timbre') 
    binary_tags.remove('gender') 

    for index, row in df_binary.iterrows():
        new_row = {'song': str(index)}

        #getting tags from binary dataframe if confidence is above threshold, otherwise 'uncertain'
        for tag in binary_tags:
            confidence_tag = f"confidence_{tag}"
            if row[confidence_tag] >= confidences[tag]:
                new_row[tag] = row[tag]
            else:
                new_row[tag] = 'uncertain'
        rows_binary.append(new_row)

    df_tags = pd.DataFrame(rows_binary)

    # #-------------------------- Resolve conflicting labels ------------------------------------#
    #the models might predict two opposite tags for the same song, both with high confidence.
    # Aggressive vs Relaxed
    mask = (df_tags["mood_aggressive"] == "aggressive") & (df_tags["mood_relaxed"] == "relaxed")
    df_tags.loc[mask, ["mood_aggressive", "mood_relaxed"]] = "uncertain"

    # Happy vs Sad 
    mask = (df_tags["mood_happy"] == "happy") & (df_tags["mood_sad"] == "sad")
    df_tags.loc[mask, ["mood_happy", "mood_sad"]] = "uncertain"

    # acustic vs electronic
    mask = (df_tags["mood_acoustic"] == "acoustic") & (df_tags["mood_electronic"] == "electronic")
    df_tags.loc[mask, ["mood_acoustic", "mood_electronic"]] = "uncertain"

    #------ return final results -------#
    return df_tags

def create_triplets(tags_df, split):
    tags_df['song'] = 'original/' + tags_df['song'] + '.mp3' #so the idx are in the 'save name'
    tags_df = tags_df.set_index('song')
    columns_tags = tags_df.columns.tolist()

    #---------------------------------- create the matrix of embeddings to use in knn-------------------------------------
    all_songs = tags_df.index.tolist()
    print(f"Total unique songs: {len(all_songs)}")

    embeddings_sim_matrix = []
    for song in all_songs:
        embedding_sim_path = os.path.join(MUSICNN_EMBEDDINGS_FOLDER, song.replace('.mp3', '.npy'))
        embedding_sim = np.load(embedding_sim_path).squeeze()  # shape [D,]
        embeddings_sim_matrix.append(embedding_sim)

    #normalize all embeddings
    embeddings_sim_matrix = normalize(np.array(embeddings_sim_matrix, dtype=np.float32), axis=1)
    print(f"shape of embeddings_sim_matrix: {np.array(embeddings_sim_matrix).shape}")

    #take the cosine similarity of the matrix with itself to get the similarity between all songs
    similarity_matrix = cosine_similarity(embeddings_sim_matrix)

    #check range of similarity values
    print(f"similarity matrix range: min={similarity_matrix.min()}, max={similarity_matrix.max()}")

    #convert similarity to percentage (0 to 100)
    sim_percent = ((similarity_matrix + 1) / 2) * 100

    #check range of similarity percentage values
    print(f"similarity percentage range: min={sim_percent.min()}, max={sim_percent.max()}")


    #--------------------------------------------generate triplets---------------------------------------------------------#
    triplets_rows = []
    similarity_rows = []

    for i, reference_song in enumerate(all_songs):
        reference_song_features = tags_df.loc[reference_song]

        print(f"Processing song {i+1}/{len(all_songs)}")

        # Get indices of songs within a threshold of similarity
        sim_percent_row = sim_percent[i]

        #Build list of similar songs at least threshold% similarity and same genre
        threshold = 90
        candidate_indices = np.where(sim_percent_row >= threshold)[0]

        # sort by similarity (descending)
        candidate_indices = sorted(
            candidate_indices,
            key=lambda idx: sim_percent_row[idx],
            reverse=True
        )

        all_similar_songs = [all_songs[idx] for idx in candidate_indices]

        similarity_new_row = {
            'reference_song': reference_song,
            'similar_songs': all_similar_songs,
        }
        similarity_rows.append(similarity_new_row)
        
        print(f"using {len(all_similar_songs)} similar songs for reference song {reference_song}")

        #for each song similar to the reference song, i create triplets based on their tags
        for target_song in all_similar_songs:
            #iterate tag by tag
            for col in columns_tags:
                reference_tag = reference_song_features[col]

                if reference_tag == 'uncertain':
                    continue

                target_song_features = tags_df.loc[target_song]
                target_tag = target_song_features[col]

                if target_tag == 'uncertain':
                    continue

                if reference_tag != target_tag:

                    modification = get_modification_from_tag(reference_tag)

                    modification_text = augment_modification(modification)
                    
                    # Create a new row for the triplet
                    new_row = {
                        'reference_song': reference_song,
                        'modification_text': modification_text,
                        'target_song': target_song,
                        'modification':modification
                    }

                    triplets_rows.append(new_row)

    # Save the triplets DataFrame to a CSV file
    triplets_df = pd.DataFrame(triplets_rows)
    triplets_df.to_csv(os.path.join(TRIPLETS_FOLDER, f"triplets_mtiq_{split}.csv"), index=False)

    #save similarity df
    similarity_df = pd.DataFrame(similarity_rows)
    similarity_df.to_csv(os.path.join(TRIPLETS_FOLDER, f"similarity_mtiq_{split}.csv"), index=False)

def main():
    binary_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'classifications/song_labels_binary.csv'))

    #split into train, val and test
    print("Splitting dataset into train, val and test...")
    df_splits = split_train_test(binary_df, train_ratio=0.6, val_ratio=0.2)
    df_splits.to_csv(os.path.join(TRIPLETS_FOLDER, 'splits.csv'), index=False)

    #for each song, select tags with confidence above threshold, otherwise 'uncertain'
    print("Selecting tags based on confidence thresholds...")
    tags_df = select_tags(binary_df)
    tags_df.to_csv(os.path.join(TRIPLETS_FOLDER, 'labels.csv'), index=False)

    #for each split, create triplets based on the selected tags and the similarity of the embeddings.
    #merge tags_df with df_splits to get the split for each song
    print("Creating triplets for each split...")
    df_tags_splits = pd.merge(tags_df, df_splits, on='song')
    for split in ['val', 'test', 'train']:
        df = df_tags_splits[df_tags_splits['split'] == split]
        create_triplets(df, split=split)

if __name__ == "__main__":
    main()
