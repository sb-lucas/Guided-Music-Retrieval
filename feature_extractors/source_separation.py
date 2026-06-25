import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import demucs.separate

def separate_audio(song_name):
    try:
        demucs.separate.main([
            "-o", "path/to/separated_audios",  # Output folder for separated audio
            song_name
        ])
        # Return success message to the main thread
        return f"Success: {os.path.basename(song_name)}"
    except Exception as e:
        return f"Error on {os.path.basename(song_name)}: {str(e)}"

def main():
    # --- 1. Parallel Audio Processing ---
    jamendo_path = 'path/to/audios/'

    audio_files = []
    for i, subfolder in enumerate(os.listdir(jamendo_path)):
        subfolder_path = os.path.join(jamendo_path, subfolder)
        for j, file in enumerate(os.listdir(subfolder_path)):
            if file.endswith('.mp3'):
                song_name = os.path.join(subfolder_path, file)
                audio_files.append(song_name)

    tam_audio = len(audio_files)
    max_workers = 8

    print("Starting audio extraction...")
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the pool
            futures_audio = [
                executor.submit(separate_audio, song_name) 
                for song_name in audio_files
            ]
            # Track progress as they complete
            for i, future in enumerate(as_completed(futures_audio)):
                try:
                    result = future.result()
                    print(f"{i+1}/{tam_audio} | {result}")
                except Exception as e:
                    print(f"FAILED on audio {i + 1}. Error: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user. Shutting down.")
        # Cancel pending tasks + stop accepting new ones
        executor.shutdown(wait=False, cancel_futures=True)

        # Force kill remaining child processes
        for p in multiprocessing.active_children():
            p.terminate()

        print("All workers terminated.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()