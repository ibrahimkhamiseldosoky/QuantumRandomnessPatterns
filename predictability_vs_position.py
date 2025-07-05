import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
with open("training_sequences.txt") as f:
    sequences = [line.strip() for line in f if len(line.strip()) == 128]

sequences = [[int(b) for b in seq] for seq in sequences]

accuracies = []

# Loop over target bit positions from 1 to 127
for target_pos in range(1, 128):
    X = [seq[:target_pos] for seq in sequences]
    y = [seq[target_pos] for seq in sequences]

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pad sequences to length 127 for consistent input size
    def pad(arr): return np.pad(arr, (0, 127 - len(arr)), 'constant')

    X_train = np.stack([pad(row) for row in X_train])
    X_test = np.stack([pad(row) for row in X_test])

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)

    # Simple MLP
    model = nn.Sequential(
        nn.Linear(127, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCELoss()

    for epoch in range(5):  # Keep it fast
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy().flatten()
        preds = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test_tensor.numpy(), preds)
        accuracies.append(acc)

    print(f"Bit {target_pos:3d}: Accuracy = {acc:.4f}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(range(1, 128), accuracies, label="Predictability", color="blue")
plt.axhline(0.5, linestyle="--", color="gray", label="Random baseline (50%)")
plt.title("Bit Index vs Predictability")
plt.xlabel("Bit Position (Target Bit)")
plt.ylabel("Prediction Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predictability_vs_bit_position.png", dpi=300)
plt.show()
