# Epileptic Seizure Classifier

A deep learning project for **EEG-based epileptic seizure classification** using **Transformer Encoders** and **Time2Vec** temporal embeddings.  
This model learns to detect seizure activity from EEG signals by analyzing their **bandpower features** across multiple frequency bands.

## Overview

- **Objective:** Automatically classify EEG signals into seizure and non-seizure categories.  
- **Input:** EEG epochs with extracted **bandpower features** (δ, θ, α, β, γ).  
- **Output:** Seizure state classification (multi-class).  
- **Core Idea:** Combine **spectral bandpower features** with **temporal embeddings (Time2Vec)** and a **Transformer Encoder** to model both frequency and temporal dependencies.


## Model Architecture

EEG Bandpower Features
↓
Linear Projection
↓
Time2Vec Encoding
↓
Transformer Encoder (Multi-Head Attention)
↓
Mean Pooling
↓
Fully Connected Layer
↓
Softmax Output

## Dataset

**Temple University Hospital EEG Corpus (TUH EEG)**  
Used subsets:  
- **TUSZ (Temple University Seizure Corpus)**  
- **TUH EEG v2.0.0**

🔗 Dataset Link: [TUH EEG Corpus Download Page](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml)

