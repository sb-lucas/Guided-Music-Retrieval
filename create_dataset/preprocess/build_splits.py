import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT_FOLDER = os.getenv("ROOT_FOLDER")

SPLIT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtg-jamendo_split-0")
OUTPUT_PATH = os.path.join(ROOT_FOLDER, "data", "splits.csv")

#official MTG-Jamendo files (autotagging task, split 0) and the name each one gets in splits.csv
SPLIT_FILES = {
    "train": "autotagging-train.tsv",
    "val": "autotagging-validation.tsv",
    "test": "autotagging-test.tsv",
}

def read_tracks(tsv_path):
    #the tags are tab separated, so the rows have a variable number of fields. Only the first four are needed
    tracks = []
    with open(tsv_path, "r") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            artist_id, path = parts[1], parts[3]
            song = os.path.splitext(path)[0]  # '99/311399.mp3' -> '99/311399'
            tracks.append({"song": song, "artist": artist_id})

    return pd.DataFrame(tracks)

def main():
    split_dfs = []
    for split, file_name in SPLIT_FILES.items():
        df = read_tracks(os.path.join(SPLIT_FOLDER, file_name))
        df["split"] = split
        split_dfs.append(df)

    all_df = pd.concat(split_dfs, ignore_index=True)

    #sanity checks: a song must belong to a single split, and the official splits are artist-disjoint
    assert not all_df["song"].duplicated().any(), "a song appears more than once"
    artist_splits = all_df.groupby("artist")["split"].nunique()
    assert (artist_splits == 1).all(), "an artist appears in more than one split"

    for split in SPLIT_FILES:
        print(f"{split}: {(all_df['split'] == split).sum()} songs, {all_df[all_df['split'] == split]['artist'].nunique()} artists")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_df[["song", "split"]].to_csv(OUTPUT_PATH, index=False)
    print(f"saved {len(all_df)} songs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
