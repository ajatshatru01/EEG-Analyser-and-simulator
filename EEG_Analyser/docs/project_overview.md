# Project Overview : EEG Analyser

## Motivation

The human brain produces complex electrical signals measurable through **Electroencephalography (EEG)**. These signals are essential for neuroscience, neurology, and brain–computer interface research but are often noisy, non-stationary, and highly individual, making analysis difficult.

Our motivation lies in two key areas:

1. **Accurate EEG Analysis** — Enhancing the extraction of meaningful EEG features and reliable brain-state classification using advanced deep learning methods.
2. **Simulation of Brain States** — Building a system that can simulate changes in EEG features under different conditions, such as drug effects or mental state variations.

By combining analysis and simulation, this project aims to create a practical, educational, and research-oriented tool that bridges the gap between raw EEG data and interpretable insights.

---

## Abstract

Electroencephalography (EEG) is a non-invasive technique for recording brain activity, widely used in neurological diagnosis and mental-state studies. However, due to the non-stationary and artifact-prone nature of EEG, effective analysis remains challenging.

This project, titled **“EEG Data Analyser and Simulator,”** introduces a two-phase framework:

- **EEG Analyser:** Handles preprocessing, feature extraction (band power, Hjorth parameters, entropy), and classification using deep learning models.
- **EEG Simulator:** Creates a digital twin of EEG activity to model changes in brain states through controlled feature modifications.

By merging analysis with simulation, the system enhances interpretability, supports research applications, and sets the foundation for personalized digital brain models.

---

## Introduction

### About EEG

EEG records the brain’s electrical activity using scalp electrodes, capturing synchronized neuron firing patterns. It is widely used for studying cognition, detecting neurological disorders, and powering Brain–Computer Interface (BCI) systems. Its high temporal resolution and affordability make it a preferred method in both clinical and research domains.

### Challenges in EEG Analysis

EEG signals often contain noise and artifacts from eye movements, muscle activity, or external interference. They also vary significantly across individuals, making generalization difficult. Traditional feature-based approaches may fail to represent the dynamic and nonlinear nature of brain activity, calling for more robust computational techniques.

### Our Objective

The goal of the EEG Data Analyser and Simulator is to design a unified system that:

- **Analyses EEG data** through preprocessing, feature extraction, and deep learning classification.
- **Simulates EEG feature changes** to visualize how brain states evolve under various interventions.
