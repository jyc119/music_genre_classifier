"""

genres original - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous
GTZAN dataset, the MNIST of sounds)


images original - A visual representation for each audio file. One way to classify data is through neural networks.
Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files
were converted to Mel Spectrograms to make this possible.


2 CSV files - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance
computed over multiple features that can be extracted from an audio file. The other file has the same structure, but
the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into
our classification models). With data, more is always better.


"""
import os

import librosa
import librosa.display
import numpy as np
import pandas as pd
import time
import models

# GENRES = 'blues classical country disco hiphop'.split()
GENRES = set('blues classical country disco hiphop jazz metal pop reggae rock'.split())
ROOT_DIR = 'data/genres_original'


def extract_features(file, chunk_duration=3):
    # y, sr = librosa.load(file, duration=30)  # Load full 30s file

    try:
        y, sr = librosa.load(file, duration=30)
    except Exception as e:
        print(f"Failed to load {file}: {e}")
        return

    samples_per_chunk = int(chunk_duration * sr)
    total_chunks = len(y) // samples_per_chunk

    all_records = []
    feature_names = None

    for i in range(total_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        y_chunk = y[start:end]

        # Skip silent chunks (optional)
        if np.sum(np.abs(y_chunk)) < 0.01:
            continue

        # --- Mel Spectrogram ---
        mel = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)
        mel_var = np.var(mel_db, axis=1)

        # --- Spectral Centroid ---
        spectral_centroid = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)
        centroid_mean = np.mean(spectral_centroid)
        centroid_var = np.var(spectral_centroid)

        # --- Tempo ---
        tempo, _ = librosa.beat.beat_track(y=y_chunk, sr=sr)

        # --- MFCCs (13–20) ---
        mfcc = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=20)
        mfcc_means = [np.mean(mfcc[i]) for i in range(13, 20)]
        mfcc_vars = [np.var(mfcc[i]) for i in range(13, 20)]

        # Combine features
        features = np.hstack([
            mel_mean, mel_var,
            centroid_mean, centroid_var,
            tempo,
            mfcc_means, mfcc_vars
        ])

        if feature_names is None:
            feature_names = (
                    [f'mel_mean_{i}' for i in range(len(mel_mean))] +
                    [f'mel_var_{i}' for i in range(len(mel_var))] +
                    ['centroid_mean', 'centroid_var', 'tempo'] +
                    [f'mfcc{i + 1}_mean' for i in range(13, 20)] +
                    [f'mfcc{i + 1}_var' for i in range(13, 20)]
            )

        all_records.append((features, feature_names))

    return all_records


# def all_features():
#     records = []
#     feature_names = None
#
#     for genre in GENRES:
#         genre_path = os.path.join(ROOT_DIR, genre)
#         for filename in os.listdir(genre_path):
#             if filename.endswith('.wav'):
#                 file_path = os.path.join(genre_path, filename)
#                 track_id = os.path.splitext(filename)[0]  # e.g., 'blues.00005'
#
#                 chunked_records = extract_features(file_path)
#
#                 for i, (features, names) in enumerate(chunked_records):
#                     if feature_names is None:
#                         feature_names = names + ['genre', 'track_id', 'chunk_index']
#                     record = dict(zip(names, features))
#                     record['genre'] = genre
#                     record['track_id'] = track_id
#                     record['chunk_index'] = i
#                     records.append(record)
#
#     df = pd.DataFrame(records, columns=feature_names)
#     return df

def all_features():
    records = []
    feature_names = None

    for genre in os.listdir(ROOT_DIR):
        if genre not in GENRES:
            continue
        genre_path = os.path.join(ROOT_DIR, genre)
        for filename in os.listdir(genre_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(genre_path, filename)
                track_id = os.path.splitext(filename)[0]

                chunked_records = extract_features(file_path)

                if chunked_records is None:
                    print(f"Skipping bad file: {file_path}")
                    continue

                for features, names in chunked_records:
                    if feature_names is None:
                        feature_names = names + ['genre', 'track_id']
                    record = dict(zip(names, features))
                    record['genre'] = genre
                    record['track_id'] = track_id
                    records.append(record)

    df = pd.DataFrame(records, columns=feature_names)
    return df


def main():
    start = time.time()
    features = all_features()
    models.train_knn_chunk_level(features)
    models.train_knn_with_majority(features)
    models.train_rf(features)
    models.train_rf_full(features)
    models.train_mlp(features)
    # end = time.time()
    # print("Time taken: " + str(end - start)) 63


if __name__ == '__main__':
    main()
