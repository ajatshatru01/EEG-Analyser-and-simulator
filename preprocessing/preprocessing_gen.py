import mne
import numpy as np
from mne.time_frequency import psd_array_welch
from antropy import hjorth_params
from antropy.entropy import spectral_entropy, perm_entropy

# 1. Load raw EEG file

raw = mne.io.read_raw_edf("file_name.edf", preload=True)

# 2. Keep only EEG channels

raw.pick_types(eeg=True)


# 3. Apply montage (standard 10-20)

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing='ignore')

print("Final EEG Channels:", raw.ch_names)

# 4. Extract raw data

data, times = raw.get_data(return_times=True)  # shape = (n_channels, n_samples)
sfreq = raw.info['sfreq']
n_channels, n_samples = data.shape
print("Data shape:", data.shape)

# 5. Define windowing

window_sec = 2  # 2s windows
window_size = int(window_sec * sfreq)
n_windows = n_samples // window_size

# 6. Feature extraction per window

features = []

for w in range(n_windows):
    start = w * window_size
    stop = start + window_size
    window_data = data[:, start:stop]

    window_features = []

    for ch in range(n_channels):
        x = window_data[ch]

        # --- Power Spectral Density (bandpower in 7 bands) ---
        psd, freqs = psd_array_welch(x, sfreq=sfreq, fmin=0.5, fmax=40, n_fft=256)
        psd_bands = []
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 25), (25, 30), (30, 40)]
        for fmin, fmax in bands:
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            psd_bands.append(np.mean(psd[idx]))

        # --- Hjorth Parameters ---
        hjorth = hjorth_params(x)

        # --- Entropy Features ---
        spec_ent = spectral_entropy(x, sfreq, method='welch', normalize=True)
        perm_ent = perm_entropy(x, normalize=True)

        # combine channel features
        ch_features = psd_bands + list(hjorth) + [spec_ent, perm_ent]
        window_features.extend(ch_features)

    features.append(window_features)

# 7. Convert to final array

features = np.array(features)   # (n_windows, 351)
features = np.expand_dims(features, axis=0)  # (1, n_windows, 351)

print("Final feature shape:", features.shape)
