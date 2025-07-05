import json

# Load your large dataset JSON file
with open("example.json", "r") as f:
    data = json.load(f)

# Extract samples and num_bits
samples = data["results"][0]["data"]["c"]["samples"]
num_bits = data["results"][0]["data"]["c"]["num_bits"]

# Convert hex samples to binary strings with padding
bitstrings = [format(int(s, 16), f'0{num_bits}b') for s in samples]

# Concatenate all bitstrings into one long string
all_bits = "".join(bitstrings)

# Parameters for training sequences
sequence_length = 128  # You can adjust this

# Prepare training sequences by chunking the long bitstring
sequences = []
for i in range(0, len(all_bits) - sequence_length + 1, sequence_length):
    seq = all_bits[i : i + sequence_length]
    sequences.append(seq)

# Save sequences to a file, one per line (bitstring sequences)
with open("sequences.txt", "w") as f:
    for seq in sequences:
        f.write(seq + "\n")

print(f"Saved {len(sequences)} sequences, each {sequence_length} bits long.")
