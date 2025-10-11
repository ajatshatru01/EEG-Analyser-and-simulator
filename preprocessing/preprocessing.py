import os
import numpy as np
import mne
from mne.time_frequency import psd_array_welch
from antropy import hjorth_params
from antropy.entropy import spectral_entropy, perm_entropy


def preprocess_edf(file_path, window_sec=5):
   
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Pick only EEG channels
        raw.pick_types(eeg=True)

        # Standard 10/20 montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        standard_channels = [
            "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
        ]

        # Ensure consistent channel order, add missing as zero
        present_channels = [ch for ch in standard_channels if ch in raw.ch_names]
        missing_channels = [ch for ch in standard_channels if ch not in raw.ch_names]

        raw.pick(present_channels)
        if missing_channels:
            zero_data = np.zeros((len(missing_channels), raw.n_times))
            info_missing = mne.create_info(missing_channels, raw.info['sfreq'], ch_types="eeg")
            raw.add_channels([mne.io.RawArray(zero_data, info_missing)])

        raw.reorder_channels(standard_channels)

        data, _ = raw.get_data(return_times=True)
        sfreq = raw.info["sfreq"]
        n_channels, n_samples = data.shape

        # Sliding windows
        window_size = int(window_sec * sfreq)
        n_windows = n_samples // window_size

        # EEG frequency bands
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 16),
                 (16, 25), (25, 30), (30, 40)]
        band_indices = None

        features = []
        for w in range(n_windows):
            start, stop = w * window_size, (w + 1) * window_size
            window_data = data[:, start:stop]

            # PSD
            psd, freqs = psd_array_welch(window_data, sfreq=sfreq,
                                         fmin=0.5, fmax=40, n_fft=256, n_jobs=1)

            if band_indices is None:
                band_indices = [np.logical_and(freqs >= fmin, freqs <= fmax)
                                for fmin, fmax in bands]

            psd_bands = np.stack([psd[:, idx].mean(axis=1) for idx in band_indices], axis=1)

            window_features = []
            for ch in range(n_channels):
                x = window_data[ch]
                ch_psd = psd_bands[ch].tolist()
                hjorth = hjorth_params(x)
                spec_ent = spectral_entropy(x, sfreq, method="welch", normalize=True)
                perm_ent = perm_entropy(x, normalize=True)

                ch_features = ch_psd + list(hjorth) + [spec_ent, perm_ent]
                window_features.extend(ch_features)

            features.append(window_features)

        return np.array(features)

    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None


def preprocess_dataset(root_folder, folder_label_pairs, window_sec=5, max_files_per_class=60):
   
    all_X, all_y = [], []

    for folder_name, label in folder_label_pairs:
        folder_path = os.path.join(root_folder, folder_name)
        count = 0
        print(f"\nScanning folder: {folder_name}")

        for dirpath, _, filenames in os.walk(folder_path):
            for f in filenames:
                if f.lower().endswith(".edf"):
                    file_path = os.path.join(dirpath, f)
                    print(f" Processing: {file_path}")

                    X_trial = preprocess_edf(file_path, window_sec)
                    if X_trial is not None and X_trial.shape[0] > 0:
                        all_X.append(X_trial)
                        all_y.append(label)
                        count += 1

                    if count >= max_files_per_class:
                        break
            if count >= max_files_per_class:
                break

    if not all_X:
        raise ValueError("No valid EDF files found.")

    # Padding to equal length
    max_windows = max(x.shape[0] for x in all_X)
    n_features = all_X[0].shape[1]
    X_final = np.zeros((len(all_X), max_windows, n_features))
    valid_lengths = []

    for i, x in enumerate(all_X):
        X_final[i, :x.shape[0], :] = x
        valid_lengths.append(x.shape[0])

    y = np.array(all_y)

    # Normalize across whole dataset
    mean = X_final.mean(axis=(0, 1), keepdims=True)
    std = X_final.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X_final - mean) / std
    X = np.nan_to_num(X, nan=0.0)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y, valid_lengths = X[indices], y[indices], np.array(valid_lengths)[indices]

    print("Dataset ready:")
    print(" Shape:", X.shape, " Labels:", y.shape)
    return X, y, valid_lengths

root_folder = "/content/drive/My Drive/TUH_EEG"
folders = [("00_epilepsy", 0), ("01_no_epilepsy", 1)]

X, y, valid_lengths = preprocess_dataset(root_folder, folders, window_sec=5)
