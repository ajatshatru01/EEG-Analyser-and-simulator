# Dataset Description

Since this project focuses on post-recording EEG analysis, data is sourced from publicly available datasets such as the TUH EEG Corpus and HUP iEEG Dataset rather than live recordings.  
Each model in the project is trained on different datasets.

---

## Seizure Classification — TUH EEG Corpus

The Temple University Hospital EEG Epilepsy Corpus (TUEP) is the largest public EEG database, with over 16,000 sessions from 10,000+ patients.  
It includes 19–21 channel recordings at 250–500 Hz, stored in EDF format, and provides labels for Normal vs. Abnormal EEGs as well as the TUSZ subset for seizure detection, providing labels for Seizure vs. Non-Seizure.

---

## Emotion Classification — SEED Dataset

The SEED (SJTU Emotion EEG Dataset) contains EEG data from 15 subjects who viewed emotion-evoking film clips.  
Recordings were made using 62 channels at 1000 Hz (downsampled to 200 Hz) and include differential entropy features representing positive, neutral, and negative emotions. In this project, pre-extracted features were used for training and testing.

---

## Stress Classification — SAM40 Dataset

The SAM40 EEG Stress Dataset includes recordings from 40 healthy participants performing both stress-inducing and relaxation tasks.  
EEG was collected using a 32-channel Emotiv EPOC Flex at 128 Hz.  
Preprocessing (artifact removal and filtering) was already performed, with samples labeled as stress or non-stress depending on the task condition.

---

## Surgery Outcome Classification — HUP iEEG Dataset

The HUP iEEG Dataset contains invasive EEG data from 58 epilepsy patients who underwent surgical treatment.  
Recordings include ictal (seizure) and interictal periods, along with surgery outcomes such as “success” or “failure.”  
Our model predicts whether the surgery was successful or not by extracting various EEG features.

---

## Dream Emotion Classification — DEED Dataset

The DEED Dream EEG Dataset records EEG during REM sleep for 38 participants across multiple nights.  
After each REM phase, participants reported dream emotions (positive, neutral, or negative).  
The data predicts the emotional state of the subject from the REM-stage EEG signals.
