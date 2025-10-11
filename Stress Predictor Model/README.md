# Stress EEG Classifier

A deep learning project for **EEG-based stress level detection** using **Transformer encoders**.  
This model analyzes **bandpower, Hjorth, and statistical features** across multiple frequency bands to predict stress levels from EEG signals.

## Overview

- **Objective:** Automatically detect stress levels from EEG signals.  
- **Input:** EEG epochs with extracted **bandpower (δ, θ, α, β, γ), Hjorth parameters, and statistical features**.  
- **Output:** Stress level classification (low, medium, high).  
- **Core Idea:** Combine **spectral EEG features** with **temporal embeddings (Time2Vec)** and **Transformer encoders** to model both frequency and temporal dependencies in stress-related patterns.  
- **Handling Class Imbalance:** Used **SMOTE (Synthetic Minority Oversampling Technique)** to augment minority classes and improve model generalization.

## Model Architecture

EEG Features (Bandpower + Hjorth + Statistical Features)  
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

## Evaluation

![Stress model evaluation]()
## Dataset

**SAM40 EEG Dataset** – EEG recordings for stress recognition.  
🔗 Dataset Link: [SAM40 Dataset](https://ieee-dataport.org/documents/sam40-stress-dataset)

