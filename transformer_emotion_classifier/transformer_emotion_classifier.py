import numpy as np
import scipy.io as sio
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

folder_path = "/content/drive/My Drive/test"

trials = []
for file_path in sorted(glob.glob(os.path.join(folder_path, "*.mat"))):
    if "label" in file_path:
        continue

    mat_data = sio.loadmat(file_path)
    trial_features = []

    for i in range(1, 16):
        key = f'de_movingAve{i}'
        if key in mat_data:
            arr = mat_data[key]  # shape: (62, T, 5)
            T = arr.shape[1]
            trial_seq = arr.transpose(1, 0, 2).reshape(T, -1)  # (T, 310)
            trial_features.append(trial_seq)

    # Padding to maximum sequence length in this trial
    if len(trial_features) > 0:
        max_T = max(f.shape[0] for f in trial_features)
        padded_features = []
        for f in trial_features:
            pad_width = ((0, max_T - f.shape[0]), (0, 0))  # pad along time axis
            f_padded = np.pad(f, pad_width, mode='constant', constant_values=0)
            padded_features.append(f_padded)

        trial_features = np.stack(padded_features, axis=0)  # (15, max_T, 310)
        trial_features = np.mean(trial_features, axis=0)    # (max_T, 310)
        trials.append(trial_features)

# Padding all trials to same length (max_T across dataset)
global_max_T = max(t.shape[0] for t in trials)
padded_trials = []
for t in trials:
    pad_width = ((0, global_max_T - t.shape[0]), (0, 0))
    t_padded = np.pad(t, pad_width, mode='constant', constant_values=0)
    padded_trials.append(t_padded)

X = np.stack(padded_trials)

label_data = sio.loadmat(os.path.join(folder_path, "label.mat"))
y = label_data['label'].squeeze()
y = np.repeat(y, 3)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
print("X shape ", X.shape)
print("y shape:", y.shape)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None):
        _, _, _, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score)
        return score @ v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, _ = self.attention(q, k, v, mask=mask)
        return self.w_concat(self.concat(out))
    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        return tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, length, head * d_tensor)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) #weights
        self.beta = nn.Parameter(torch.zeros(d_model)) #bias
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, drop_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, hidden_dim, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, num_layers, drop_prob=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, hidden_dim, drop_prob)
            for _ in range(num_layers)
        ])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EEGTransformer(nn.Module):
    def __init__(self, feature_dim=310, d_model=64, n_head=2, hidden_dim=256, num_layers=1, drop_prob=0.3, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.encoder = TransformerEncoder(d_model, n_head, hidden_dim, num_layers, drop_prob)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x, mask=None, predict=False):
        x = self.input_proj(x)
        x = self.encoder(x, mask)
        x = x.mean(dim=1)  # mean pooling
        logits = self.classifier(x) #(45,3)
        if predict:
            preds = torch.argmax(logits, dim=1) #picks class of highest score so basically it will be of (45,)
            mapping = torch.tensor([-1, 0, 1], device=preds.device) #0,1,2 are converted to -1,0,1
            return mapping[preds].unsqueeze(1)
        return logits


if __name__ == "__main__":
    model = EEGTransformer()
    criterion = nn.CrossEntropyLoss() #Cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Normalize data
    X = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-8)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Remap labels to {0,1,2}
    if y.min().item() < 0:
        y_mapped = (y + 1).long()
    else:
        y_mapped = y

    # train-test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
    X, y_mapped,
    test_size=0.2,
    random_state=42,
    stratify=y_mapped
)


    # Train for 20 epochs
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_train).float().mean().item()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Accuracy: {acc:.4f}")


    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_test).float().mean().item()
        print(f"Test Accuracy: {test_acc:.4f}")

        # Accuracy
        acc = accuracy_score(y_test, test_preds)
        print("Accuracy:", acc)

        # Precision,Recall,F1
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average=None, labels=[0, 1, 2])
        print("Precision per class:", precision)
        print("Recall per class:", recall)
        print("F1-score per class:", f1)

        # Full classification report
        print("\nFull classification report:")
        print(classification_report(y_test, test_preds, labels=[0, 1, 2]))

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2])
        print("Confusion Matrix:\n", cm)






    preds = model(X, predict=True)
    print("Predictions shape:", preds.shape)
    print("Sample predictions:", preds[:10].view(-1).tolist())
    print("True labels sample:", y_mapped[:5].numpy())
    print("Predicted classes:", preds[:5].numpy())