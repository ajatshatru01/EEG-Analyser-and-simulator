import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import math
import torch.optim as optim

X = np.load("/content/drive/MyDrive/EEG_features_5s.npy")
y = np.load("/content/drive/MyDrive/EEG_labels_5s.npy")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        _, _, _, d_tensor = k.size()
        score = (q @ k.transpose(2, 3)) / math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        return self.softmax(score) @ v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_tensor = d_model // n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, _ = self.attention(q, k, v, mask)
        return self.w_concat(self.concat(out))

    def split(self, tensor):
        B, L, D = tensor.size()
        return tensor.view(B, L, self.n_head, self.d_tensor).transpose(1, 2)

    def concat(self, tensor):
        B, H, L, D = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(B, L, H * D)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, drop_prob=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, hidden_dim, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, memory_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, hidden_dim, num_layers, drop_prob=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_head, hidden_dim, drop_prob)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x

class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.randn(d_model - 1))

    def forward(self, t):
        linear_term = self.w0 * t + self.b0
        periodic = torch.sin(t * self.w + self.b)
        return torch.cat([linear_term, periodic], dim=-1)

class EEGTransformer(nn.Module):
    def __init__(self, feature_dim=228, d_model=64, n_head=4,
                 hidden_dim=256, num_layers=2, drop_prob=0.5):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.time2vec = Time2Vec(d_model)
        self.encoder = TransformerEncoder(d_model, n_head, hidden_dim, num_layers, drop_prob)
        self.decoder = TransformerDecoder(d_model, n_head, hidden_dim, num_layers, drop_prob)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, mask=None, predict=False):
        B, T, _ = x.shape
        x_proj = self.input_proj(x)
        t = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1).float()
        x_enc = x_proj + self.time2vec(t)
        enc_output = self.encoder(x_enc, mask)
        dec_input = torch.zeros(B, T, enc_output.size(-1), device=x.device)
        dec_output = self.decoder(dec_input, enc_output, None, mask)
        x = dec_output.mean(dim=1)
        logits = self.classifier(x)
        if predict:
            probs = torch.sigmoid(logits)
            return (probs >= 0.5).long()
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGTransformer(feature_dim=X.shape[2], d_model=128, n_head=8).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss, train_acc = 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        preds = (torch.sigmoid(logits) >= 0.5).long()
        acc = (preds == y_batch.long()).float().mean()
        train_loss += loss.item()
        train_acc += acc.item()

    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            acc = (preds == y_batch.long()).float().mean()
            val_loss += loss.item()
            val_acc += acc.item()

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc/len(val_loader):.4f}")
