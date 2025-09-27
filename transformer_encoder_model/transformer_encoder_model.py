

import torch
import torch.nn as nn
import torch.optim as optim
import math

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

class Time2Vec(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        self.w = nn.Parameter(torch.randn(1, d_model - 1))
        self.b = nn.Parameter(torch.randn(1, d_model - 1))

    def forward(self, t):
        # t: (batch, seq_len, 1)
        linear_term = self.w0 * t + self.b0                  # (B, T, 1)
        periodic_terms = torch.sin(self.w * t + self.b)      # (B, T, d_model-1)
        return torch.cat([linear_term, periodic_terms], dim=-1)

class EEGTransformer(nn.Module):
    def __init__(self, feature_dim=228, d_model=128, n_head=8,
                 hidden_dim=512, num_layers=4, drop_prob=0.2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.time2vec = Time2Vec(d_model=d_model)
        self.encoder = TransformerEncoder(d_model, n_head, hidden_dim, num_layers, drop_prob)
        self.classifier = nn.Linear(d_model, 1)   # binary classification

    def forward(self, x, mask=None, predict=False):
        batch_size, seq_len, _ = x.shape
        x_proj = self.input_proj(x)   
        t = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1).float()
        t2v = self.time2vec(t)       
        x = x_proj + t2v
        x = self.encoder(x, mask)    
        x = x.mean(dim=1)           
        logits = self.classifier(x)  
        if predict:
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            return preds
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

feature_dim = X.shape[2]  
model = EEGTransformer(feature_dim=feature_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


num_epochs = 25
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
