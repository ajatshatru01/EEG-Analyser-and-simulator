# Surgery EEG Classifier

A deep learning project for **EEG-based cognitive load detection during surgery tasks** using **Transformer encoders**.  
This model analyzes **bandpower and Hjorth features** across multiple frequency bands to classify different cognitive states of surgeons.

## Overview

- **Objective:** Automatically classify EEG signals into cognitive load or task-specific states during surgery.  
- **Input:** EEG epochs with extracted **bandpower (δ, θ, α, β, γ) and Hjorth features**.  
- **Output:** Cognitive/task state classification (multi-class).  
- **Core Idea:** Combine **spectral EEG features** with **temporal embeddings (Time2Vec)** and **Transformer encoders** to model both frequency and temporal dependencies in cognitive states.

## Model Architecture

EEG Features (Bandpower + Hjorth)  
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

## Model evaluation

![Surgery model validation](https://github.com/ajatshatru01/EEG-Analyser-and-simulator/blob/test/SURGERY%20PREDICTION%20MODEL/surgery_evaluation.jpeg)

## Dataset

**HUP / IEEG Dataset** – Human Intracranial EEG Data during Surgery Tasks.  
🔗 Dataset Link: [HUP/IEEG Portal](https://www.ieeg.org/)

