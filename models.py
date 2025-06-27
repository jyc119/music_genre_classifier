from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import numpy as np
from collections import Counter
import collections
import pandas as pd
import csv

from metrics import performance_metric, build_confusion_matrix
from helper import write_csv, write_csv_predictions


def evaluate_model_majority_vote(df):
    X = df.drop(columns=['genre', 'track_id'])
    y = df['genre']
    track_ids = df['track_id']

    # Split by unique tracks
    unique_tracks = df['track_id'].unique()
    track_genres = df.groupby('track_id')['genre'].first()
    train_tracks, test_tracks = train_test_split(
        unique_tracks,
        stratify=track_genres[unique_tracks],
        test_size=0.2,
        random_state=42
    )

    # Index masks
    train_mask = df['track_id'].isin(train_tracks)
    test_mask = df['track_id'].isin(test_tracks)

    # Extract splits
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    test_track_ids = track_ids[test_mask]

    return X_train, X_test, y_train, y_test, test_track_ids


def train_knn(df):
    # Split into train/test once
    X = df.drop(columns=['genre', 'track_id'])
    y = df['genre']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    best_k = None
    best_score = 0
    best_pred = None
    best_pipeline = None

    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('knn', knn)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if k == 1:
            write_csv_predictions(y_pred, y_test)
        print(f"k = {k}, Test accuracy = {accuracy:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_k = k
            best_pred = y_pred
            best_pipeline = pipeline

    print(f"\nBest k: {best_k}, Test accuracy: {best_score:.4f}")
    performance_metric(y_test, best_pred)
    build_confusion_matrix(X_test, y_test, pipeline)


def train_knn_chunk_level(df):
    # Step 1: Get unique track IDs and their genres
    unique_tracks = df['track_id'].unique()
    track_genres = df.groupby('track_id')['genre'].first()

    # Step 2: Split track IDs into train/test
    train_tracks, test_tracks = train_test_split(
        unique_tracks,
        stratify=track_genres[unique_tracks],
        test_size=0.2,
        random_state=42
    )

    # Step 3: Build masks to filter chunk-level data
    train_mask = df['track_id'].isin(train_tracks)
    test_mask = df['track_id'].isin(test_tracks)

    # Step 4: Split the actual chunk-level data
    X = df.drop(columns=['genre', 'track_id'])
    y = df['genre']
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    best_k = None
    best_score = 0
    best_pred = None
    best_pipeline = None

    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('knn', knn)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"k = {k}, Chunk-level test accuracy (no leakage): {accuracy:.4f}")

        if accuracy > best_score:
            best_score = accuracy
            best_k = k
            best_pred = y_pred
            best_pipeline = pipeline

    print(f"\nBest k: {best_k}, Best chunk-level accuracy: {best_score:.4f}")


def train_knn_with_majority(df):
    X_train, X_test, y_train, y_test, test_track_ids = evaluate_model_majority_vote(df)

    best_k = None
    best_score = 0
    best_preds = None
    best_trues = None

    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('knn', knn)
        ])

        pipeline.fit(X_train, y_train)
        chunk_preds = pipeline.predict(X_test)

        # Map predictions to track IDs
        track_pred_map = collections.defaultdict(list)
        for pred, track_id in zip(chunk_preds, test_track_ids):
            track_pred_map[track_id].append(pred)

        # if k == 1:
            # write_csv(track_pred_map)  # assuming this logs predictions per track

        final_preds = []
        final_true = []

        for track_id, preds in track_pred_map.items():
            # Count genre occurrences
            counts = collections.Counter(preds)
            most_common = counts.most_common()

            # Check if there's a majority and no tie at top
            top_genre, top_count = most_common[0]
            tied = len([count for _, count in most_common if count == top_count])

            # Use mode regardless of tie
            track_mode = collections.Counter(preds).most_common(1)[0][0]
            final_preds.append(track_mode)

            # Only for majority-voted tracks
            final_true.append(df[df['track_id'] == track_id]['genre'].iloc[0])

        # if k == 1:
            # write_csv_predictions(final_preds, final_true)
        # Compute accuracy
        score = accuracy_score(final_true, final_preds)
        print(f"k = {k}, Hybrid Test accuracy = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_preds = final_preds
            best_trues = final_true

    print(f"\nBest k: {best_k}, Test accuracy: {best_score:.4f}")

    # performance_metric(best_trues, best_preds)


def train_mlp(df):
    best_score = 0
    best_config = None

    # Step 1: Get unique track IDs and their genres
    unique_tracks = df['track_id'].unique()
    track_genres = df.groupby('track_id')['genre'].first()

    # Step 2: Split track IDs into train/test
    train_tracks, test_tracks = train_test_split(
        unique_tracks,
        stratify=track_genres[unique_tracks],
        test_size=0.2,
        random_state=42
    )

    # Step 3: Build masks to filter chunk-level data
    train_mask = df['track_id'].isin(train_tracks)
    test_mask = df['track_id'].isin(test_tracks)

    # Step 4: Split the actual chunk-level data
    X = df.drop(columns=['genre', 'track_id'])
    y = df['genre']
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Example: trying multiple hidden layer sizes
    hidden_layer_configs = [(100,), (200,), (100, 50), (128, 64)]

    for config in hidden_layer_configs:
        mlp = MLPClassifier(hidden_layer_sizes=config, max_iter=500, random_state=42)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('mlp', mlp)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        score = accuracy_score(y_test, y_pred)

        print(f"MLP {config}, CV accuracy = {score:.4f}")

        if score > best_score:
            best_score = score
            best_config = config

    print(f"\nBest config: {best_config}, CV accuracy: {best_score:.4f}")


def train_rf(df):
    # Step 1: Get unique track IDs and their genres
    unique_tracks = df['track_id'].unique()
    track_genres = df.groupby('track_id')['genre'].first()

    # Step 2: Split track IDs into train/test
    train_tracks, test_tracks = train_test_split(
        unique_tracks,
        stratify=track_genres[unique_tracks],
        test_size=0.2,
        random_state=42
    )

    # Step 3: Build masks to filter chunk-level data
    train_mask = df['track_id'].isin(train_tracks)
    test_mask = df['track_id'].isin(test_tracks)

    # Step 4: Split the actual chunk-level data
    X = df.drop(columns=['genre', 'track_id'])
    y = df['genre']
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Split into train/test once
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, stratify=y, test_size=0.2, random_state=42
    # )

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42))
    ])

    # Train and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Accuracy and report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest Non-majority Test Accuracy (chunk-level): {accuracy:.4f}")


def train_rf_full(df):
    X_train, X_test, y_train, y_test, test_track_ids = evaluate_model_majority_vote(df)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42))
    ])

    # Train and predict
    pipeline.fit(X_train, y_train)
    chunk_preds = pipeline.predict(X_test)

    # Map predictions to track IDs
    track_pred_map = collections.defaultdict(list)
    for pred, track_id in zip(chunk_preds, test_track_ids):
        track_pred_map[track_id].append(pred)

    final_preds = []
    final_true = []

    for track_id, preds in track_pred_map.items():
        # Count genre occurrences
        counts = collections.Counter(preds)
        most_common = counts.most_common()

        # Use mode regardless of tie
        track_mode = collections.Counter(preds).most_common(1)[0][0]
        final_preds.append(track_mode)

        # Only for majority-voted tracks
        final_true.append(df[df['track_id'] == track_id]['genre'].iloc[0])

    # Majority vote per track
    track_pred_map = {}
    for pred, track_id in zip(chunk_preds, test_track_ids):
        track_pred_map.setdefault(track_id, []).append(pred)

    final_preds = []
    final_true = []

    for track_id, preds in track_pred_map.items():
        vote = max(set(preds), key=preds.count)
        final_preds.append(vote)
        final_true.append(df[df['track_id'] == track_id]['genre'].iloc[0])

    # Report
    score = np.mean(np.array(final_preds) == np.array(final_true))
    print(f"RF Test accuracy = {score:.4f}")
