#!/usr/bin/env python3
"""
Quantum Transformer Test Dataset Evaluator

Loads a trained Quantum Transformer model and evaluates it on a test dataset.
Provides comprehensive performance metrics and analysis.

Usage: python test_quantum_transformer.py --test_file path/to/test_data.txt --model_path path/to/best_model.pt
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# =====================================================================================
# MODEL ARCHITECTURE (Same as training script)
# =====================================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, d_model = x.shape
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, fill_value)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(context), attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        attn_output, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x, attn_weights

class QuantumTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        x = self.norm(x)
        logits = self.output_projection(x)
        return logits, attention_weights

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# =====================================================================================
# TOKENIZER AND DATASET
# =====================================================================================

class BitTokenizer:
    def __init__(self):
        self.token_to_id = {'0': 0, '1': 1, '<PAD>': 2}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id[char] for char in text if char in self.token_to_id]

    def decode(self, token_ids: List[int]) -> str:
        return ''.join([self.id_to_token[id] for id in token_ids if id in self.id_to_token])

class TestDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: BitTokenizer, sequence_length: int):
        self.tokenizer = tokenizer
        self.sequences = []
        self.sequence_length = sequence_length
        self._load_data(file_path)
        print(f"Loaded {len(self.sequences)} test sequences")

    def _load_data(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) >= self.sequence_length:
                    # Take exactly sequence_length characters
                    sequence = line[:self.sequence_length]
                    encoded = self.tokenizer.encode(sequence)
                    if len(encoded) == self.sequence_length:
                        self.sequences.append(encoded)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]
        return input_seq, target_seq

# =====================================================================================
# EVALUATION FUNCTIONS
# =====================================================================================

def calculate_perplexity(loss: float) -> float:
    return math.exp(loss)

def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    return correct.mean().item()

def analyze_sequence_patterns(sequences: List[str]) -> Dict:
    """Analyze patterns in binary sequences"""
    analysis = {
        'bit_distribution': Counter(),
        'bigram_distribution': Counter(),
        'trigram_distribution': Counter(),
        'entropy': [],
        'run_lengths': {'0': [], '1': []},
        'sequence_lengths': []
    }
    
    for seq in sequences:
        analysis['sequence_lengths'].append(len(seq))
        analysis['bit_distribution'].update(seq)
        
        # Bigrams and trigrams
        for i in range(len(seq) - 1):
            analysis['bigram_distribution'][seq[i:i+2]] += 1
        for i in range(len(seq) - 2):
            analysis['trigram_distribution'][seq[i:i+3]] += 1
        
        # Entropy calculation
        bit_counts = Counter(seq)
        total = len(seq)
        if total > 0:
            entropy = -sum((c/total) * math.log2(c/total) for c in bit_counts.values())
            analysis['entropy'].append(entropy)
        
        # Run length analysis
        if seq:
            current_bit = seq[0]
            run_length = 1
            for bit in seq[1:]:
                if bit == current_bit:
                    run_length += 1
                else:
                    analysis['run_lengths'][current_bit].append(run_length)
                    current_bit = bit
                    run_length = 1
            analysis['run_lengths'][current_bit].append(run_length)
    
    return analysis

def evaluate_model_on_test_set(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate model performance on test set"""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_targets = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for input_ids, target_ids in tqdm(test_loader, desc="Testing"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, 3), target_ids.view(-1))  # vocab_size = 3
            
            batch_accuracy = calculate_accuracy(logits, target_ids)
            
            total_loss += loss.item()
            total_accuracy += batch_accuracy * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            # Store predictions for further analysis
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(target_ids.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_accuracy / total_samples
    perplexity = calculate_perplexity(avg_loss)
    
    return {
        'test_loss': avg_loss,
        'test_accuracy': avg_accuracy,
        'test_perplexity': perplexity,
        'predictions': all_predictions,
        'targets': all_targets,
        'total_samples': total_samples
    }

def generate_test_samples(model: nn.Module, tokenizer: BitTokenizer, device: torch.device,
                         num_samples: int = 100, max_length: int = 64) -> List[str]:
    """Generate samples using the trained model"""
    model.eval()
    samples = []
    max_context = model.max_seq_len - 1
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Start with random bit
            current_sequence = [np.random.choice([0, 1])]
            
            for _ in range(max_length - 1):
                context = current_sequence[-max_context:]
                input_tensor = torch.tensor([context], device=device)
                
                logits, _ = model(input_tensor)
                logits = logits[0, -1, :2]  # Only consider 0 and 1 tokens
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                current_sequence.append(next_token)
            
            samples.append(tokenizer.decode(current_sequence))
    
    return samples

def create_test_evaluation_plots(test_results: Dict, test_sequences: List[str], 
                               generated_sequences: List[str], save_dir: str):
    """Create comprehensive evaluation plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Test Dataset Evaluation Results', fontsize=16, fontweight='bold')
    
    # Test metrics
    metrics = ['Test Loss', 'Test Accuracy', 'Test Perplexity']
    values = [test_results['test_loss'], test_results['test_accuracy'], test_results['test_perplexity']]
    
    axes[0, 0].bar(metrics, values, color=['red', 'green', 'blue'], alpha=0.7)
    axes[0, 0].set_title('Test Performance Metrics')
    axes[0, 0].set_ylabel('Value')
    for i, v in enumerate(values):
        axes[0, 0].text(i, v + max(values) * 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Prediction vs Target distribution
    pred_counter = Counter(test_results['predictions'])
    target_counter = Counter(test_results['targets'])
    
    labels = ['0', '1', 'PAD']
    pred_values = [pred_counter.get(i, 0) for i in range(3)]
    target_values = [target_counter.get(i, 0) for i in range(3)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, target_values, width, label='Targets', alpha=0.7)
    axes[0, 1].bar(x + width/2, pred_values, width, label='Predictions', alpha=0.7)
    axes[0, 1].set_title('Token Distribution: Targets vs Predictions')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].legend()
    
    # Analyze test sequences
    test_analysis = analyze_sequence_patterns(test_sequences)
    gen_analysis = analyze_sequence_patterns(generated_sequences)
    
    # Entropy comparison
    axes[0, 2].hist(test_analysis['entropy'], alpha=0.7, label='Test Data', bins=20)
    axes[0, 2].hist(gen_analysis['entropy'], alpha=0.7, label='Generated', bins=20)
    axes[0, 2].set_title('Entropy Distribution Comparison')
    axes[0, 2].set_xlabel('Entropy')
    axes[0, 2].legend()
    
    # Bit distribution comparison
    test_bits = [test_analysis['bit_distribution'].get('0', 0), test_analysis['bit_distribution'].get('1', 0)]
    gen_bits = [gen_analysis['bit_distribution'].get('0', 0), gen_analysis['bit_distribution'].get('1', 0)]
    
    x = ['0', '1']
    width = 0.35
    x_pos = np.arange(len(x))
    
    axes[1, 0].bar(x_pos - width/2, test_bits, width, label='Test Data', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, gen_bits, width, label='Generated', alpha=0.7)
    axes[1, 0].set_title('Bit Distribution Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(x)
    axes[1, 0].legend()
    
    # Run length comparison
    test_runs_0 = test_analysis['run_lengths']['0']
    test_runs_1 = test_analysis['run_lengths']['1']
    gen_runs_0 = gen_analysis['run_lengths']['0']
    gen_runs_1 = gen_analysis['run_lengths']['1']
    
    if test_runs_0 and test_runs_1 and gen_runs_0 and gen_runs_1:
        axes[1, 1].hist([test_runs_0, test_runs_1], alpha=0.7, label=['Test 0-runs', 'Test 1-runs'], bins=20)
        axes[1, 1].hist([gen_runs_0, gen_runs_1], alpha=0.7, label=['Gen 0-runs', 'Gen 1-runs'], bins=20)
        axes[1, 1].set_title('Run Length Distribution')
        axes[1, 1].set_xlabel('Run Length')
        axes[1, 1].legend()
    
    # Sample sequences display
    sample_text = f"Test Sample Sequences:\n"
    for i, seq in enumerate(test_sequences[:5]):
        sample_text += f"{i+1}: {seq[:50]}...\n"
    
    sample_text += f"\nGenerated Sample Sequences:\n"
    for i, seq in enumerate(generated_sequences[:5]):
        sample_text += f"{i+1}: {seq[:50]}...\n"
    
    axes[1, 2].text(0.05, 0.95, sample_text, transform=axes[1, 2].transAxes,
                   fontsize=8, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Sample Sequences')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test Quantum Transformer on test dataset')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test dataset file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--num_generated_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🧪 QUANTUM TRANSFORMER TEST EVALUATION")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    vocab_size = config.get('VOCAB_SIZE', 3)
    d_model = config.get('D_MODEL', 192)
    n_heads = config.get('N_HEADS', 6)
    n_layers = config.get('N_LAYERS', 4)
    d_ff = config.get('D_FF', 768)
    sequence_length = config.get('SEQUENCE_LENGTH', 128)
    dropout = config.get('DROPOUT', 0.1)
    
    # Initialize model
    model = QuantumTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=sequence_length,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully! Parameters: {model.count_parameters():,}")
    
    # Initialize tokenizer and dataset
    tokenizer = BitTokenizer()
    test_dataset = TestDataset(args.test_file, tokenizer, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate model
    print("\n📊 Evaluating model performance...")
    test_results = evaluate_model_on_test_set(model, test_loader, device)
    
    # Generate samples for comparison
    print(f"\n🎲 Generating {args.num_generated_samples} samples...")
    generated_sequences = generate_test_samples(model, tokenizer, device, args.num_generated_samples)
    
    # Get test sequences for comparison
    test_sequences = [tokenizer.decode(test_dataset[i][1].tolist()) for i in range(min(100, len(test_dataset)))]
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Test Loss:      {test_results['test_loss']:.4f}")
    print(f"Test Accuracy:  {test_results['test_accuracy']:.4f}")
    print(f"Test Perplexity: {test_results['test_perplexity']:.4f}")
    print(f"Total Test Samples: {test_results['total_samples']:,}")
    
    # Analyze patterns
    print("\n📈 Analyzing sequence patterns...")
    test_analysis = analyze_sequence_patterns(test_sequences)
    gen_analysis = analyze_sequence_patterns(generated_sequences)
    
    print(f"\nEntropy Comparison:")
    print(f"  Test Data:  Mean={np.mean(test_analysis['entropy']):.4f}, Std={np.std(test_analysis['entropy']):.4f}")
    print(f"  Generated:  Mean={np.mean(gen_analysis['entropy']):.4f}, Std={np.std(gen_analysis['entropy']):.4f}")
    
    print(f"\nBit Distribution:")
    test_bits = test_analysis['bit_distribution']
    gen_bits = gen_analysis['bit_distribution']
    print(f"  Test Data:  0: {test_bits.get('0', 0)}, 1: {test_bits.get('1', 0)}")
    print(f"  Generated:  0: {gen_bits.get('0', 0)}, 1: {gen_bits.get('1', 0)}")
    
    # Create visualizations
    print(f"\n📊 Creating evaluation plots...")
    create_test_evaluation_plots(test_results, test_sequences, generated_sequences, args.output_dir)
    
    # Save detailed results
    detailed_results = {
        'test_metrics': {
            'test_loss': test_results['test_loss'],
            'test_accuracy': test_results['test_accuracy'],
            'test_perplexity': test_results['test_perplexity'],
            'total_samples': test_results['total_samples']
        },
        'pattern_analysis': {
            'test_data': {
                'entropy_mean': float(np.mean(test_analysis['entropy'])),
                'entropy_std': float(np.std(test_analysis['entropy'])),
                'bit_distribution': dict(test_analysis['bit_distribution']),
                'top_bigrams': test_analysis['bigram_distribution'].most_common(10),
                'top_trigrams': test_analysis['trigram_distribution'].most_common(10)
            },
            'generated_data': {
                'entropy_mean': float(np.mean(gen_analysis['entropy'])),
                'entropy_std': float(np.std(gen_analysis['entropy'])),
                'bit_distribution': dict(gen_analysis['bit_distribution']),
                'top_bigrams': gen_analysis['bigram_distribution'].most_common(10),
                'top_trigrams': gen_analysis['trigram_distribution'].most_common(10)
            }
        },
        'model_info': {
            'parameters': model.count_parameters(),
            'architecture': config
        }
    }
    
    results_file = os.path.join(args.output_dir, 'test_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("✨ Test evaluation complete!")

if __name__ == "__main__":
    main()