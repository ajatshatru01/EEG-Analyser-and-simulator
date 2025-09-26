from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import glob
import os

folder_path = "/content/drive/My Drive/test"

trials = []
for file_path in sorted(glob.glob(os.path.join(folder_path, "*.mat"))):
    if "label" in file_path:
        continue

    mat_data = sio.loadmat(file_path)
    trial_features = []

    for i in range(1, 16):
        key = f'de_movingAve{i}'
        if key in mat_data:
            arr = mat_data[key] # shape: (62, T, 5)
            T = arr.shape[1]
            trial_seq = arr.transpose(1, 0, 2).reshape(T, -1) # (T, 310)
            trial_features.append(trial_seq)

    if len(trial_features) > 0:
        max_T = max(f.shape[0] for f in trial_features)
        padded_features = []
        for f in trial_features:
            pad_width = ((0, max_T - f.shape[0]), (0, 0))
            f_padded = np.pad(f, pad_width, mode='constant', constant_values=0)
            padded_features.append(f_padded)

        trial_features = np.stack(padded_features, axis=0) # (15, max_T, 310)
        trial_features = np.mean(trial_features, axis=0) # (max_T, 310)
        trials.append(trial_features)

global_max_T = max(t.shape[0] for t in trials)
padded_trials = []
for t in trials:
    pad_width = ((0, global_max_T - t.shape[0]), (0, 0))
    t_padded = np.pad(t, pad_width, mode='constant', constant_values=0)
    padded_trials.append(t_padded)

X = np.stack(padded_trials)

label_data = sio.loadmat(os.path.join(folder_path, "label.mat"))
y = label_data['label'].squeeze()
y = np.repeat(y, 3)
y=y+1

print("Original X shape:", X.shape) # (45, T, 310)
print("Original y shape:", y.shape)

def sliding_window(X, y, window_size, step_size):
    new_X, new_y = [], []
    for i in range(len(X)):
        seq = X[i]
        label = y[i]
        T = seq.shape[0]
        for start in range(0, T - window_size + 1, step_size): #sliding window over timesteps
            end = start + window_size
            new_X.append(seq[start:end])
            new_y.append(label)
    return np.array(new_X), np.array(new_y)

window_size = 100 #timesteps per window
step_size = 50 #50% overlap

X_aug, y_aug = sliding_window(X, y, window_size, step_size)

print("Augmented X shape:", X_aug.shape) # (n_windows, 200, 310)
print("Augmented y shape:", y_aug.shape)

X_aug = (X_aug - np.mean(X_aug, axis=(0, 1), keepdims=True)) / (np.std(X_aug, axis=(0, 1), keepdims=True) + 1e-8)


X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug,
    test_size=0.2,
    random_state=42,
    stratify=y_aug
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f" — test_accuracy: {acc:.4f}")

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=regularizers.l2(1e-5)))
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(1e-5)))
model.add(Dropout(0.1))
model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-5)))
model.add(Dropout(0.1))
model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=8,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[early_stop, TestAccuracyCallback(X_test, y_test)],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {acc*100:.2f}%')
