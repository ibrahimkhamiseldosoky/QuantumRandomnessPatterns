import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load data
with open("training_sequences.txt", "r") as f:
    lines = [line.strip() for line in f if len(line.strip()) == 128]

X = []
y = []
for seq in lines:
    input_bits = [int(b) for b in seq[:30]]  # bits 0–29
    X.append(input_bits[:29])                # 0–28 → input
    y.append(input_bits[29])                 # bit 29 → target

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)

# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(29, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Train
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.6f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).numpy().flatten()
    preds_binary = (preds > 0.5).astype(int)
    y_true = y_test_tensor.numpy().flatten().astype(int)

acc = accuracy_score(y_true, preds_binary)
print(f"\n🎯 Prediction Accuracy for Bit 29 (using bits 0–28): {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_true, preds_binary))
