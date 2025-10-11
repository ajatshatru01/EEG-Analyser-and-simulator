# Emotion EEG Classifier

A deep learning project for **EEG-based emotion recognition (SEED dataset)** using **LSTM networks**.  
This model analyzes **bandpower, Hjorth, and statistical features** across multiple frequency bands to classify emotional states.

## Overview

- **Objective:** Automatically classify EEG signals into emotional states.  
- **Input:** EEG epochs with extracted **bandpower (δ, θ, α, β, γ), Hjorth parameters, and statistical features**.  
- **Output:** Emotion classification (multi-class).  
- **Core Idea:** Use **temporal modeling via LSTMs** to capture sequential EEG patterns associated with emotions.  
- **Handling Class Imbalance:** Used **SMOTE (Synthetic Minority Oversampling Technique)** to augment minority classes.

## Model Architecture

EEG Features (Bandpower + Hjorth + Statistical Features)  
↓  
LSTM Layer 1 (128 units, return sequences=True)  
↓  
LSTM Layer 2 (64 units, return sequences=False)  
↓  
Dense Layer 1 (64 units, ReLU)  
↓  
Dense Layer 2 (32 units, ReLU)  
↓  
Softmax Output

## Evaluation

![Emotion model evaluation](https://github.com/ajatshatru01/EEG-Analyser-and-simulator/blob/test/emotion-prediction-model/EMOTION.jpeg)
## Dataset

**SEED EEG Dataset** – EEG recordings for emotion recognition.  
🔗 Dataset Link: [SEED Dataset](http://bcmi.sjtu.edu.cn/~seed/SEED.html)

