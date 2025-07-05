import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter

def load_sequences(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if len(line.strip()) == 128]

def calculate_entropy(sequences):
    def entropy(p):
        if p == 0 or p == 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    entropies = []
    for i in range(128):
        bits = [seq[i] for seq in sequences]
        count = Counter(bits)
        p1 = count.get('1', 0) / len(bits)
        entropies.append(entropy(p1))
    return entropies

# Load both datasets
brisbane_sequences = load_sequences("training_sequences.txt")
torino_sequences = load_sequences("torino_test_sequences.txt")

# Calculate entropy
brisbane_entropy = calculate_entropy(brisbane_sequences)
torino_entropy = calculate_entropy(torino_sequences)

# Plotting
plt.figure(figsize=(12, 5))
plt.plot(brisbane_entropy, label="IBM Brisbane", color="blue", linewidth=2)
plt.plot(torino_entropy, label="IBM Torino", color="red", linewidth=2, linestyle='--')
plt.title("Entropy per Bit Index (Quantum Collapse Sequence)")
plt.xlabel("Bit Index (0 = First Collapsed Qubit)")
plt.ylabel("Entropy (bits)")
plt.ylim(0.996, 1.001)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy_comparison_brisbane_vs_torino.png", dpi=300)
plt.show()
