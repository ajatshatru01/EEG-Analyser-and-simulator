# Preprocessing

The given **EEG Data Preprocessing Pipeline** is a general pipeline used in this project that prepares raw EEG recordings for feature extraction and model analysis. It standardizes, cleans, and segments the input signals to ensure consistency and computational accuracy.

---

## Workflow Overview

### 1. Loading Raw Data

EEG signals are loaded from **.edf (European Data Format)** files using the **MNE library**. Data is preloaded into memory to enable smooth and efficient processing.

### 2. Channel Selection

Only **EEG channels** are retained, while non-brain signals such as EOG, EMG, and ECG are filtered out to focus purely on neural activity.

### 3. Montage Application

A **standard 10–20 electrode montage** is applied to map electrode positions accurately. This ensures spatial consistency and comparability across datasets.

### 4. Data Extraction

The pipeline extracts EEG signals, timestamps, and metadata such as **sampling rate**, **number of channels**, and **total samples**, which are required for later feature extraction.

### 5. Signal Windowing

The continuous EEG data is divided into **fixed-length segments** (e.g., 2-second windows). This segmentation allows localized analysis of temporal changes and improves efficiency for feature-level computations.
