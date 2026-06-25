import os
import numpy as np
import pandas as pd

def read_tsv(file_path):
    data = []
    with open(file_path, 'r') as f:
        # Read header
        header_line = f.readline().strip()
        header = header_line.split('\t')[:5] + ['TAGS']  # Rename last field if needed

        # Read the rest of the file line by line
        for line in f:
            parts = line.strip().split('\t')
            fixed = parts[:5]  # First five columns
            tags = parts[5:]   # Tags start from the 6th column
            tags = [tag.split('---')[1] for tag in tags if '---' in tag]
            fixed.append(', '.join(tags))  # Join all tag names
            data.append(fixed)

    df = pd.DataFrame(data, columns=header)

    df['DURATION'] = pd.to_numeric(df['DURATION'], errors='coerce')  # handles non-numeric values safely
    df['TAGS'] = df['TAGS'].apply(lambda x: [tag.strip() for tag in x.split(',')] if isinstance(x, str) else [])

    return df
