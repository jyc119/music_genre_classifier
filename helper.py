import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter
import os
import soundfile as sf
from pydub import AudioSegment

def get_audio_signal(file):
    y, sr = librosa.load(file)

    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title('Audio Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def plot_spectrogram(file):
    y, sr = librosa.load(file)

    # Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (STFT)')
    plt.tight_layout()
    plt.show()

def plot_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=15)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def plot_spectral_centroid(file_path):
    y, sr = librosa.load(file_path, duration=15)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroids))
    times = librosa.frames_to_time(frames, sr=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(times, spectral_centroids, color='r', label='Spectral Centroid')
    plt.title('Spectral Centroid Over Time')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=15)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

def plot_audio_features(file_path):
    y, sr = librosa.load(file_path, duration=15)

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(centroid))
    centroid_times = librosa.frames_to_time(frames, sr=sr)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Mel Spectrogram
    librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axs[0], cmap='magma')
    axs[0].set_title('Mel Spectrogram')
    fig.colorbar(axs[0].images[0], ax=axs[0], format='%+2.0f dB')

    # Spectral Centroid
    librosa.display.waveshow(y, sr=sr, alpha=0.4, ax=axs[1])
    axs[1].plot(centroid_times, centroid, color='r', label='Spectral Centroid')
    axs[1].set_title('Spectral Centroid Over Time')
    axs[1].legend()

    # MFCC
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axs[2], cmap='coolwarm')
    axs[2].set_title('MFCC')
    fig.colorbar(axs[2].images[0], ax=axs[2])

    plt.tight_layout()
    plt.show()

def write_csv(data):
    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Values...", "Full Value List", "Counts", "Majority"])

        for key, values in data.items():
            vote = max(set(values), key=values.count)
            counts = dict(Counter(values))  # Use if you care about frequency
            writer.writerow([key] + values + [str(counts)] + [vote])


def write_csv_predictions(predictions, true_values, filename="preds.csv"):
    if len(predictions) != len(true_values):
        raise ValueError("Arrays must be of equal length")

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["prediction", "true_value"])

        for pred, true in zip(predictions, true_values):
            writer.writerow([pred, true])


def draw_graph():
    training = [0.582, 0.769, 0.8432, 0.9201, 0.9484,
                0.9687,0.989, 0.9922, 0.9940, 0.9966,
                0.9968, 0.9976, 0.9981, 0.9983, 0.9983,
                0.9984, 0.9984, 0.9985, 0.9986, 0.9986]

    validation = [0.69, 0.72, 0.77, 0.815, 0.805,
                  0.8, 0.835, 0.805, 0.81, 0.825,
                  0.845, 0.81, 0.83, 0.835, 0.83,
                  0.835, 0.82, 0.825, 0.83, 0.83]

    training_cnn = [0.5623854441414458, 0.7778175153113684, 0.8707586391881622, 0.9244043095355179, 0.9374580893200412,
                    0.9570834637221154, 0.9662032276811658, 0.9664267513076132, 0.9723724797711119, 0.9714336805400331,
                    0.973, 0.975, 0.974, 0.974, 0.976,
                    0.976, 0.9765, 0.976, 0.977, 0.978]

    validation_cnn = [0.67, 0.79, 0.795, 0.795, 0.83,
                      0.805, 0.795, 0.785, 0.795, 0.805,
                      0.78, 0.785, 0.78, 0.8, 0.8,
                      0.805, 0.795, 0.78, 0.79, 0.805,
                      ]

    # epochs = range(1, len(training) + 1)
    #
    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs, training, label='Training Accuracy', marker='o')
    # plt.plot(epochs, validation, label='Validation Accuracy', marker='s')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training vs Validation Accuracy over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.xticks(range(1, 21))
    # plt.show()

    epochs = range(1, 21)

    plt.figure(figsize=(10, 6))

    # CNN+RNN
    plt.plot(epochs, training, label='CNN+RNN Training', marker='o', linestyle='-', color='#1f77b4')
    plt.plot(epochs, validation, label='CNN+RNN Validation', marker='s', linestyle='--', color='#1f77b4')

    # CNN
    plt.plot(epochs, training_cnn, label='CNN Training', marker='^', linestyle='-', color='#ff7f0e')
    plt.plot(epochs, validation_cnn, label='CNN Validation', marker='d', linestyle='--', color='#ff7f0e')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN vs CNN+RNN Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(epochs)
    plt.ylim(0.5, 1.1)

    plt.show()


# file = "data/genres_original/blues/blues.00000.wav"
# # plot_mel_spectrogram(file)
# # plot_spectral_centroid(file)
# # plot_mfcc(file)
# plot_audio_features(file)

draw_graph()