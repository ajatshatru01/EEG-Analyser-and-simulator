# Challenges Faced

## Large Dataset Sizes

Full EEG datasets, particularly the TUH EEG dataset with over 10,000 files (~30 GB), were too large to load entirely into memory at once. This required careful data handling and processing in batches to avoid memory overflow.

## Class Imbalance

Certain datasets, such as SAM40, exhibited a significant class imbalance (ratio ~1:3), which posed challenges for training unbiased models and necessitated techniques like oversampling or weighted loss functions.

## Limited Data Availability

Access to EEG data is often restricted due to privacy laws such as HIPAA. This scarcity of data limited the diversity and size of datasets available for model development.

## Unlabeled Data

Many datasets lacked proper labeling, requiring manual separation and annotation of data segments to create usable labels for supervised learning.

---

# Solutions

## Data Management and Augmentation

To handle large datasets, we reduced the size of the data loaded at a time and applied windowing techniques to augment the dataset, allowing the model to learn from smaller, overlapping segments efficiently.

## Addressing Class Imbalance

The class imbalance issue was mitigated by implementing SMOTE (Synthetic Minority Over-sampling Technique), which generated synthetic samples for minority classes to improve model performance.

## Data Acquisition Strategies

Due to limited publicly available EEG data, we sourced additional datasets from research papers and reached out to hospitals and research organizations, ensuring access to diverse and relevant EEG recordings.

## Accurate Data Labeling

Unlabeled data were carefully grouped and annotated by our team based on findings from relevant research papers, ensuring high-quality labels for supervised learning.
