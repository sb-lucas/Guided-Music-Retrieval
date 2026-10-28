#----------------------------------------------------------------------------------------------------
# # Pass-rate verification of the MusicGen generations used in MTSQ
#
# For every selected instrument and mood we generate new tracks the same way `triplets_instrumentation_add.py` and `triplets_mood.py` do,
# and score **only the generated audio** with the Essentia classifiers. The pass rate is the fraction of generated tracks in which the requested instrument or mood is detected.
#
# - **Instruments:** MusicGen-melody is conditioned on the `other` stem (Demucs) of a random original song and the prompt `isolated {instrument} melody`.
#   The output goes through Demucs again and its `other` stem is the *isolated generated track* that gets scored.
# - **Moods:** MusicGen is prompted with `A {mood} {genre} {subgenre} song with only {instruments}`, using the tags of a random instrumental song. The generated track is scored.
# - Detection rates of the same classifier on **real tracks** are computed as a base rate, since some tags (e.g. piano) are very common anyway.
#
# Run in the `musicgen` kernel. Everything is resumable: files that already exist are skipped.
#----------------------------------------------------------------------------------------------------

import os, sys, time, random
from pathlib import Path

#demucs calls ffprobe/ffmpeg, which live in the bin folder of this environment
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

import numpy as np
import pandas as pd
import torch
import torchaudio
from dotenv import load_dotenv, find_dotenv
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from demucs.audio import save_audio

load_dotenv(find_dotenv(usecwd=True))
ROOT_FOLDER = Path(os.getenv("ROOT_FOLDER"))
AUDIOS_FOLDER = Path(os.getenv("AUDIOS_FOLDER"))
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")
CLASSIFICATIONS_FOLDER = Path(os.getenv("CLASSIFICATIONS_FOLDER"))

sys.path.append(str(ROOT_FOLDER / "create_dataset" / "MTSQ"))
sys.path.append(str(ROOT_FOLDER / "feature_extraction" / "extractors"))
import json, datetime, audiocraft
from triplets_instrumentation_add import instrument_families, instrument_families_thresholds, get_unpresent_instruments
from triplets_mood import mood_families, mood_families_thresholds, get_song_tags, get_unpresent_moods
from demucs_separator import Demucs
from essentia_classifiers import EssentiaModels

#------------------------------------------ parameters ------------------------------------------#
N_SONGS = 1000             #songs per experiment. Quick check first (it prints the time per generation), then set to 1000
SEED = 0
MIN_STEM_RMS = 1e-3      #an 'other' stem with a lower rms (about -60 dBFS) is considered empty
INSTRUMENTAL_CONFIDENCE = 0.6   #same threshold as triplets_mood.py
KEEP_RAW = False         #keep the raw MusicGen output of the instruments, before the second demucs pass

INSTRUMENTS = ['saxophone', 'flute', 'piano', 'acoustic guitar', 'violin']
INSTRUMENT_TAG = {'acoustic guitar': 'acousticguitar'}   #name used in the prompt -> name of the essentia tag
MOODS = list(mood_families.keys())

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]    #activation thresholds used for the pass rates

OUT = Path.cwd()
STEMS_DIR = OUT / "stems"
GENERATED_DIR = OUT / "generated"
SCORES_DIR = OUT / "scores"
for d in [STEMS_DIR, GENERATED_DIR, SCORES_DIR]:
    d.mkdir(exist_ok=True)

print("outputs at:", OUT)

demucs_model = Demucs()

musicgen = MusicGen.get_pretrained('facebook/musicgen-melody')
musicgen.set_generation_params(duration=10.0)

essentia = EssentiaModels(WEIGHTS_PATH)


#----- classifications, loaded the same way as in the triplet scripts -----#
binary_df = pd.read_csv(CLASSIFICATIONS_FOLDER / "song_labels_binary.csv", index_col=0)

multilabel_df = pd.read_csv(CLASSIFICATIONS_FOLDER / "song_labels_multilabel.csv", index_col=0).set_index('song')

instrument_df = multilabel_df.drop(columns=[col for col in multilabel_df.columns if col.startswith('theme_')])
instrument_df.columns = instrument_df.columns.str.replace('^instrument_', '', regex=True)

mood_df = multilabel_df.drop(columns=[col for col in multilabel_df.columns if col.startswith('instrument_')])
mood_df.columns = mood_df.columns.str.replace('^theme_', '', regex=True)

genre_df = pd.read_csv(CLASSIFICATIONS_FOLDER / "song_genre.csv").set_index('song')

instrumental_songs = binary_df[(binary_df['voice_instrumental'] == 'instrumental') &
                               (binary_df['confidence_voice_instrumental'] > INSTRUMENTAL_CONFIDENCE)].index.tolist()

print(f"{len(binary_df)} songs, {len(instrumental_songs)} instrumental")


#----------------------------------------------------------------------------------------------------
# ## 1. Songs for the instruments: random originals with a valid `other` stem
#
# Demucs separates random original songs (all stems are saved). Songs whose `other` stem is empty are skipped and replaced by new random songs, until there are `N_SONGS` valid ones.
#----------------------------------------------------------------------------------------------------

def stem_path(song, stem):
    return STEMS_DIR / demucs_model.model_name / song.split('/')[-1] / f"{stem}.wav"

def rms(path):
    audio, _ = torchaudio.load(path)
    return audio.pow(2).mean().sqrt().item()

candidates = sorted(binary_df.index.tolist())
random.Random(SEED).shuffle(candidates)

instrument_songs, empty_stem_songs, other_rms = [], [], {}
t0 = time.time()
for song in candidates:
    if len(instrument_songs) >= N_SONGS:
        break

    other_path = stem_path(song, 'other')
    if not other_path.exists():
        demucs_model.separate_to_folder(str(AUDIOS_FOLDER / "original" / f"{song}.mp3"), str(STEMS_DIR))

    other_rms[song] = rms(other_path)
    if other_rms[song] > MIN_STEM_RMS:
        instrument_songs.append(song)
    else:
        empty_stem_songs.append(song)

print(f"{len(instrument_songs)} valid songs, {len(empty_stem_songs)} skipped for an empty 'other' stem ({time.time() - t0:.0f}s)")
pd.DataFrame({'song': instrument_songs, 'other_rms': [other_rms[s] for s in instrument_songs]}).to_csv(OUT / "instrument_songs.csv", index=False)
pd.DataFrame({'song': empty_stem_songs, 'other_rms': [other_rms[s] for s in empty_stem_songs]}).to_csv(OUT / "instrument_songs_skipped.csv", index=False)

#would the real pipeline (get_unpresent_instruments) have used each instrument on each song? Lets the pass rates be split by admitted / not admitted
instrument_admission = pd.DataFrame({song: {i: i in get_unpresent_instruments(instrument_df.loc[song]) for i in INSTRUMENTS}
                                     for song in instrument_songs}).T
instrument_admission.index.name = 'song'
instrument_admission.to_csv(OUT / "instrument_admission.csv")
print(f"instrument admitted by the absence rule in {instrument_admission.values.mean():.0%} of the (song, instrument) pairs")


#----------------------------------------------------------------------------------------------------
# ## 2. Songs for the moods: random instrumental songs with usable tags
#
# `get_song_tags` returns `None` when a song has no instrument above the activation threshold; such songs are skipped and replaced.
#----------------------------------------------------------------------------------------------------

candidates = sorted(instrumental_songs)
random.Random(SEED).shuffle(candidates)

mood_songs, mood_song_tags, no_tags_songs = [], {}, []
for song in candidates:
    if len(mood_songs) >= N_SONGS:
        break
    tags = get_song_tags(instrument_df.loc[song], mood_df.loc[song], genre_df.loc[song])
    if tags is None:
        no_tags_songs.append(song)
        continue
    mood_songs.append(song)
    mood_song_tags[song] = tags

print(f"{len(mood_songs)} instrumental songs, {len(no_tags_songs)} skipped for having no instrument tags")
pd.Series(mood_songs, name='song').to_csv(OUT / "mood_songs.csv", index=False)
pd.Series(no_tags_songs, name='song').to_csv(OUT / "mood_songs_skipped.csv", index=False)

def mood_prompt(mood, tags):
    genre, subgenre, instruments = tags
    return f"A {mood} {genre} {subgenre} song with only {', '.join(instruments)}"

#prompts used for every (song, mood), and whether the real pipeline (get_unpresent_moods) would have used that mood on that song
mood_prompts = pd.DataFrame([{'song': song, 'mood': mood, 'genre': mood_song_tags[song][0], 'subgenre': mood_song_tags[song][1],
                              'instruments': ', '.join(mood_song_tags[song][2]), 'prompt': mood_prompt(mood, mood_song_tags[song]),
                              'admitted': mood in get_unpresent_moods(mood_df.loc[song])}
                             for song in mood_songs for mood in MOODS])
mood_prompts.to_csv(OUT / "mood_prompts.csv", index=False)
print(f"mood admitted by the absence rule in {mood_prompts['admitted'].mean():.0%} of the (song, mood) pairs")


#----------------------------------------------------------------------------------------------------
# ## Run configuration
#
# Saved next to the results, to know exactly how they were produced.
#----------------------------------------------------------------------------------------------------

run_config = {
    'date': datetime.datetime.now().isoformat(timespec='seconds'),
    'n_songs': N_SONGS, 'seed': SEED, 'min_stem_rms': MIN_STEM_RMS, 'instrumental_confidence': INSTRUMENTAL_CONFIDENCE,
    'instruments': INSTRUMENTS, 'instrument_tag': INSTRUMENT_TAG, 'moods': MOODS,
    'instrument_families': instrument_families, 'instrument_families_thresholds': instrument_families_thresholds,
    'mood_families': mood_families, 'mood_families_thresholds': mood_families_thresholds,
    'musicgen': {'model': 'facebook/musicgen-melody', 'duration': 10.0, 'other_generation_params': 'audiocraft defaults', 'sample_rate': musicgen.sample_rate},
    'demucs': {'model': demucs_model.model_name, 'shifts': 1, 'overlap': 0.25},
    'essentia_weights': sorted(f for f in os.listdir(WEIGHTS_PATH) if 'discogs-effnet' in f),
    'versions': {'torch': torch.__version__, 'audiocraft': audiocraft.__version__},
}
with open(OUT / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=1)


#----------------------------------------------------------------------------------------------------
# ## 3. Generate the instruments
#
# All requested instruments of a song are generated in one batch, sharing the melody of the song. The raw MusicGen output goes through Demucs, and the `other` stem is saved as the isolated generated track.
#----------------------------------------------------------------------------------------------------

def generated_path(kind, requested, song):
    return GENERATED_DIR / kind / requested.replace(' ', '_') / f"{song.split('/')[-1]}.wav"

torch.manual_seed(SEED)
n_generated, t0 = 0, time.time()

for i, song in enumerate(instrument_songs):
    todo = [instrument for instrument in INSTRUMENTS if not generated_path('instrument', instrument, song).exists()]
    if not todo:
        continue

    melody, sr = torchaudio.load(stem_path(song, 'other'))

    #--generate melody, as in triplets_instrumentation_add.py--#
    wav = musicgen.generate_with_chroma(
        descriptions=[f"isolated {instrument} melody" for instrument in todo],
        melody_wavs=melody.expand(len(todo), -1, -1),
        melody_sample_rate=sr
    )

    for instrument, one_wav in zip(todo, wav):
        final_path = generated_path('instrument', instrument, song)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        raw_base = final_path.with_name(final_path.stem + "_raw")

        #loudness normalization at -14 LUFS, as in the triplet scripts
        audio_write(str(raw_base), one_wav.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)

        #the output may contain more instruments than the requested one, the requested one is in the 'other' stem
        stems = demucs_model.separate(str(raw_base) + ".wav")
        save_audio(stems['other'], str(final_path), samplerate=demucs_model.samplerate)

        if not KEEP_RAW:
            os.remove(str(raw_base) + ".wav")

    n_generated += len(todo)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"song {i + 1}/{len(instrument_songs)}: {n_generated} clips, {(time.time() - t0) / n_generated:.1f}s per clip", flush=True)

print(f"done: {n_generated} new clips in {time.time() - t0:.0f}s")


#----------------------------------------------------------------------------------------------------
# ## 4. Generate the moods
#----------------------------------------------------------------------------------------------------

torch.manual_seed(SEED)
n_generated, t0 = 0, time.time()

for i, song in enumerate(mood_songs):
    todo = [mood for mood in MOODS if not generated_path('mood', mood, song).exists()]
    if not todo:
        continue

    descriptions = [mood_prompt(mood, mood_song_tags[song]) for mood in todo]
    wav = musicgen.generate(descriptions)

    for mood, one_wav in zip(todo, wav):
        final_path = generated_path('mood', mood, song)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        audio_write(str(final_path.with_suffix('')), one_wav.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)

    n_generated += len(todo)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"song {i + 1}/{len(mood_songs)}: {n_generated} clips, {(time.time() - t0) / n_generated:.1f}s per clip", flush=True)

print(f"done: {n_generated} new clips in {time.time() - t0:.0f}s")


#----------------------------------------------------------------------------------------------------
# ## 5. Score the generated audio with Essentia
#
# Every generated track gets the activations of all 40 instrument tags and all 56 mood/theme tags (same models that built the dataset).
#----------------------------------------------------------------------------------------------------

def score_files(kind, requested_list, songs, csv_path):
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    done_paths = set(done['path']) if len(done) else set()
    rows = done.to_dict('records')

    t0, n_new = time.time(), 0
    for song in songs:
        for requested in requested_list:
            path = generated_path(kind, requested, song)
            if str(path) in done_paths or not path.exists():
                continue

            embeddings = essentia.get_embeddings(str(path))
            row = {'kind': kind, 'requested': requested, 'song': song, 'path': str(path)}
            for label, score in zip(*essentia.get_multiclass_labels(embeddings, 'instrument')):
                row['inst_' + label] = float(score)
            for label, score in zip(*essentia.get_multiclass_labels(embeddings, 'moodtheme')):
                row['mood_' + label] = float(score)
            rows.append(row)
            n_new += 1

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"{kind}: {n_new} new files scored, {len(rows)} in total ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)

instrument_scores = score_files('instrument', INSTRUMENTS, instrument_songs, SCORES_DIR / "generated_instrument_scores.csv")
mood_scores = score_files('mood', MOODS, mood_songs, SCORES_DIR / "generated_mood_scores.csv")


#----------------------------------------------------------------------------------------------------
# ## Load the saved results
#
# Everything above is saved in this folder: `scores/*.csv` (all activations of every generated clip), the songs and the admission flags, the mood prompts and `run_config.json`.
# To re-analyse without generating or scoring again, run only the setup cell, the classifications cell, this cell and the analysis cells below.
#----------------------------------------------------------------------------------------------------

instrument_scores = pd.read_csv(SCORES_DIR / "generated_instrument_scores.csv")
mood_scores = pd.read_csv(SCORES_DIR / "generated_mood_scores.csv")

instrument_songs = pd.read_csv(OUT / "instrument_songs.csv")['song'].tolist()
mood_songs = pd.read_csv(OUT / "mood_songs.csv")['song'].tolist()
instrument_admission = pd.read_csv(OUT / "instrument_admission.csv", index_col='song')
mood_prompts = pd.read_csv(OUT / "mood_prompts.csv")

all_instrument_labels = [c[len('inst_'):] for c in instrument_scores.columns if c.startswith('inst_')]
mood_labels = [c[len('mood_'):] for c in mood_scores.columns if c.startswith('mood_')]
print(f"{len(instrument_scores)} generated instrument clips, {len(mood_scores)} generated mood clips")


#----------------------------------------------------------------------------------------------------
# ## 6. Instrument pass rates
#
# Columns: `top1` / `top3` = the requested tag is the highest / among the 3 highest instrument activations;
# `top1_family` = the highest tag belongs to the instrument's family in `instrument_families`; `>=t` = the requested tag has an activation of at least `t`.
# `base_*` are the same rates computed on all real tracks.
#----------------------------------------------------------------------------------------------------

#song_labels_multilabel.csv has 39 instrument columns (no 'voice'). The same 39 are used for the generated audio,
#so that the ranks of generated and real tracks are comparable
instrument_labels = [label for label in all_instrument_labels if label in instrument_df.columns]
print(f"{len(instrument_labels)} instrument tags used for the ranks; left out: {sorted(set(all_instrument_labels) - set(instrument_labels))}")

def instrument_metrics(scores, tag, family_tags):
    #scores: dataframe with one column per instrument tag
    ranks = scores[instrument_labels].rank(axis=1, ascending=False, method='min')[tag]
    top1_tag = scores[instrument_labels].idxmax(axis=1)

    metrics = {
        'n': len(scores),
        'mean_score': scores[tag].mean(),
        'top1': (ranks == 1).mean(),
        'top3': (ranks <= 3).mean(),
        'top1_family': top1_tag.isin(family_tags).mean(),
    }
    for t in THRESHOLDS:
        metrics[f'>={t}'] = (scores[tag] >= t).mean()
    return metrics

generated_inst = instrument_scores.rename(columns={'inst_' + l: l for l in all_instrument_labels})

rows_generated, rows_real = {}, {}
for instrument in INSTRUMENTS:
    tag = INSTRUMENT_TAG.get(instrument, instrument)
    family_tags = sorted({t for tags in instrument_families.values() if tag in tags for t in tags})

    rows_generated[instrument] = instrument_metrics(generated_inst[generated_inst['requested'] == instrument], tag, family_tags)
    rows_real[instrument] = instrument_metrics(instrument_df, tag, family_tags)

instrument_pass_rates = pd.DataFrame(rows_generated).T
instrument_base_rates = pd.DataFrame(rows_real).T.add_prefix('base_')
instrument_report = pd.concat([instrument_pass_rates, instrument_base_rates.drop(columns='base_n')], axis=1)
instrument_report.to_csv(OUT / "pass_rates_instruments.csv")

#which of the 5 requested instruments scores highest, per requested instrument (rows sum to 1)
tags_5 = [INSTRUMENT_TAG.get(i, i) for i in INSTRUMENTS]
best_of_5 = generated_inst[tags_5].idxmax(axis=1)
confusion = pd.crosstab(generated_inst['requested'].map(lambda i: INSTRUMENT_TAG.get(i, i)), best_of_5, normalize='index')


#----------------------------------------------------------------------------------------------------
# ## 7. Mood pass rates
#
# The score of a mood is the **maximum activation over the tags of its family** (`mood_families`).
# `top1` = that mood family has the highest score among the 7 families; `>=t` = the family score is at least `t`.
# `base_*` are the same rates on real instrumental tracks.
#----------------------------------------------------------------------------------------------------

def family_scores(scores):
    #scores: dataframe with one column per mood tag
    return pd.DataFrame({mood: scores[tags].max(axis=1) for mood, tags in mood_families.items()})

def mood_metrics(fam_scores, mood):
    metrics = {
        'n': len(fam_scores),
        'mean_score': fam_scores[mood].mean(),
        'top1': (fam_scores.idxmax(axis=1) == mood).mean(),
    }
    for t in THRESHOLDS:
        metrics[f'>={t}'] = (fam_scores[mood] >= t).mean()
    return metrics

generated_mood = mood_scores.rename(columns={'mood_' + l: l for l in mood_labels})
generated_fam = family_scores(generated_mood)
real_fam = family_scores(mood_df.loc[instrumental_songs])

rows_generated = {mood: mood_metrics(generated_fam[generated_mood['requested'] == mood], mood) for mood in MOODS}
rows_real = {mood: mood_metrics(real_fam, mood) for mood in MOODS}

mood_pass_rates = pd.DataFrame(rows_generated).T
mood_base_rates = pd.DataFrame(rows_real).T.add_prefix('base_')
mood_report = pd.concat([mood_pass_rates, mood_base_rates.drop(columns='base_n')], axis=1)
mood_report.to_csv(OUT / "pass_rates_moods.csv")

#which mood family scores highest, per requested mood (rows sum to 1)
confusion = pd.crosstab(generated_mood['requested'], generated_fam.idxmax(axis=1), normalize='index')


