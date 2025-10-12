# EEG Feature Extraction Pipeline

## Why Feature Extraction?

EEG signals are complex, non-stationary, and noisy. Direct analysis of raw data is inefficient.  
Feature extraction transforms EEG signals into compact, meaningful representations that capture brain activity patterns such as frequency power, signal complexity, and regularity.

These features are crucial for tasks like classification, emotion recognition, mental state analysis, and anomaly detection.

---

## Pipeline Overview

Features are extracted from preprocessed EEG data using spectral and nonlinear measures. Signals are divided into short time windows, and features are computed for each channel independently.

### Feature Extraction Flow

**Windowing**

EEG signals are divided into fixed-length windows (e.g., 2 seconds). Each window is analyzed independently to capture time-varying brain activity.

```mermaid
for w in range(n_windows):
    start = w * window_size
    stop = start + window_size
    window_data = data[:, start:stop]
```

**Power Spectral Density (PSD) Features**

PSD shows energy distribution across frequency bands. Standard EEG bands analyzed:

- Delta (0.5–4 Hz) :-- deep sleep
- Theta (4–8 Hz) :-- drowsiness
- Alpha (8–12 Hz) :-- relaxed wakefulness
- Low Beta (12–16 Hz), High Beta (16–25 Hz) :-- alertness, active thinking
- Gamma (25–40 Hz) :-- high-level cognitive functions

Mean PSD of each band is stored as a frequency-domain feature.

```mermaid
psd, freqs = psd_array_welch(x, sfreq=sfreq, fmin=0.5, fmax=40, n_fft=256)
bands = [(0.5,4), (4,8), (8,12), (12,16), (16,25), (25,30), (30,40)]
```

**Hjorth Parameters**

Includes three time-domain features summarizing EEG signal dynamics:

- Activity – signal power
- Mobility – mean frequency
- Complexity – change in frequency content (signal irregularity)

```mermaid
hjorth = hjorth_params(x)
```

**Entropy Features**

Entropy features in EEG feature extraction measure the randomness or complexity of an EEG signal using nonlinear measures.

- Spectral Entropy – uniformity of power across frequencies (higher = more complex).

- Permutation Entropy – measures unpredictability and nonlinear dynamics.

```mermaid
spec_ent = spectral_entropy(x, sfreq, method='welch', normalize=True)
perm_ent = perm_entropy(x, normalize=True)
```

### **Combining Features**

Combine PSD, Hjorth, and entropy features for each channel into a single vector. Features from all channels in a window are stored for further analysis and model training.

```mermaid
ch_features = psd_bands + list(hjorth) + [spec_ent, perm_ent]
window_features.extend(ch_features)
```

### **Final Feature Array**

Combine features from all windows and channels into a final matrix (n_windows × n_features). This matrix is the input matrix which is ready for classification tasks.

```mermaid
features = np.array(features)
```

This EEG feature extraction pipeline converts raw time-series signals into structured numerical features representing both frequency content and signal complexity.
