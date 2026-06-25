import os
import librosa
import soundfile as sf


input_root = "../path/to/complete/audios"
output_root = "../path/to/trimmed/audios"
target_duration = 10.0  # seconds

os.makedirs(output_root, exist_ok=True)

for subdir in os.listdir(input_root):
    subdir_path = os.path.join(input_root, subdir)
    for file in os.listdir(subdir_path):
        input_file_path = os.path.join(subdir_path, file)
        print(f"Processing: {input_file_path}")

        output_subdir_path = os.path.join(output_root, subdir)
        os.makedirs(output_subdir_path, exist_ok=True)
        output_file_path = os.path.join(output_subdir_path, file)

        if os.path.exists(output_file_path):
            print(f"Already exists: {output_file_path}, skipping.")
            continue

        # Load the full audio
        y, sr = librosa.load(input_file_path, sr=None)

        total_duration = len(y) / sr
        if total_duration <= target_duration:
            # Audio shorter than 10s → skip
            print(f"Audio shorter than {target_duration}s, skipping: {input_file_path}")
            continue
        else:
            mid_point = len(y) // 2
            half_target = int((target_duration / 2) * sr)
            start = max(0, mid_point - half_target)
            end = min(len(y), mid_point + half_target)
            y_out = y[start:end]

        # Save cropped audio with the same format
        sf.write(output_file_path, y_out, sr)
        print(f"Saved: {output_file_path}")
