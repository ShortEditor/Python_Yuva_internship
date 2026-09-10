"""
Week 5 - Deep Learning Application in Data Science
Handwritten digit classification using a Convolutional Neural Network (CNN)
Dataset: scikit-learn Digits dataset (public dataset originally from UCI ML Repository)
Framework: PyTorch
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1) Load and normalize the 8x8 grayscale images
digits = load_digits()
X = digits.images.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

# 2) Stratified train/validation/test split: 64% / 16% / 20%
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20,
    stratify=y_train_full, random_state=SEED
)

def loader(Xa, ya, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(Xa[:, None, :, :], dtype=torch.float32),
            torch.tensor(ya, dtype=torch.long)
        ),
        batch_size=64, shuffle=shuffle
    )

train_loader = loader(X_train, y_train, True)
val_loader = loader(X_val, y_val)
test_loader = loader(X_test, y_test)

# 3) CNN architecture
class DigitsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = DigitsCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

def evaluate(dl):
    model.eval()
    loss_total, pred, true = 0.0, [], []
    with torch.no_grad():
        for xb, yb in dl:
            out = model(xb)
            loss = criterion(out, yb)
            loss_total += loss.item() * xb.size(0)
            pred.extend(out.argmax(1).cpu().numpy())
            true.extend(yb.cpu().numpy())
    return loss_total / len(dl.dataset), np.array(true), np.array(pred)

# 4) Training with validation monitoring and early stopping
best_state, best_val_loss = None, float("inf")
history = []
patience, bad_epochs = 6, 0

for epoch in range(1, 31):
    model.train()
    loss_total, pred, true = 0.0, [], []

    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        loss_total += loss.item() * xb.size(0)
        pred.extend(out.argmax(1).detach().cpu().numpy())
        true.extend(yb.cpu().numpy())

    train_loss = loss_total / len(train_loader.dataset)
    train_acc = accuracy_score(true, pred)
    val_loss, val_true, val_pred = evaluate(val_loader)
    val_acc = accuracy_score(val_true, val_pred)

    history.append([epoch, train_loss, val_loss, train_acc, val_acc])

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break

model.load_state_dict(best_state)

# 5) Final test evaluation
test_loss, y_true, y_pred = evaluate(test_loader)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

print("Test accuracy:", accuracy_score(y_true, y_pred))
print("Weighted precision:", precision)
print("Weighted recall:", recall)
print("Weighted F1:", f1)
print("Test loss:", test_loss)
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

pd.DataFrame(
    history, columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
).to_csv("training_history.csv", index=False)
