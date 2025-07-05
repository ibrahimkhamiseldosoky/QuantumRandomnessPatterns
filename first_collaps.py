import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy, chisquare

INPUT_FILE = "training_sequences.txt"

def load_first_bits(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    first_bits = [line.strip()[0] for line in lines if len(line.strip()) > 0]
    return first_bits

def calculate_entropy(bits):
    counts = Counter(bits)
    total = len(bits)
    probs = [count / total for count in counts.values()]
    return entropy(probs, base=2)

def chi_square_test(bits):
    counts = Counter(bits)
    observed = [counts.get("0", 0), counts.get("1", 0)]
    expected = [len(bits)/2, len(bits)/2]
    chi2_stat, p_val = chisquare(f_obs=observed, f_exp=expected)
    return chi2_stat, p_val, observed

def plot_distribution(bits):
    counts = Counter(bits)
    labels = ["0", "1"]
    values = [counts.get("0", 0), counts.get("1", 0)]
    plt.bar(labels, values, color=["skyblue", "lightgreen"])
    plt.title("Distribution of First Bits")
    plt.ylabel("Count")
    plt.savefig("first_bit_distribution.png")
    plt.close()

# Main
first_bits = load_first_bits(INPUT_FILE)

print(f"📄 Loaded {len(first_bits)} first-bit samples from {INPUT_FILE}")

ent = calculate_entropy(first_bits)
chi2_stat, p_val, observed = chi_square_test(first_bits)

print("\n🧪 First Bit Randomness Analysis")
print("================================")
print(f"Entropy (bits): {ent:.6f} / 1.0")
print(f"Counts: 0 → {observed[0]}, 1 → {observed[1]}")
print(f"Chi-square: {chi2_stat:.6f}")
print(f"P-value: {p_val:.6f}")
print("Result:", "❌ Not uniform" if p_val < 0.05 else "✅ Likely uniform (random)")

plot_distribution(first_bits)
print("\n📊 Saved distribution plot to 'first_bit_distribution.png'")
