import os
import pandas as pd

def save_results_csv(csv_path, rows, key_columns=('run', 'dataset', 'split')):
    """
    rows: list of dicts, one per model, with the information of the model (name, seed, dataset, ...) and its scores
    If the file already exists, the rows with the same key_columns replace the old ones and the other rows are kept.
    """
    csv_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    results = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        old_results = pd.read_csv(csv_path)
        results = pd.concat([old_results, results], ignore_index=True).drop_duplicates(subset=list(key_columns), keep='last')

    results.to_csv(csv_path, index=False, float_format='%.6g')
