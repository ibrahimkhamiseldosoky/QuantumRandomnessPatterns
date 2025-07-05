#!/usr/bin/env python3
"""
Quantum Bit Sequence Transformer - Character-Level GPT (Optimized for Speed & Final Analysis)
==============================================================================================

A lightweight transformer model for learning patterns in quantum bit sequences.
Designed for Kaggle environments with comprehensive logging, visualization, and final quantitative analysis.
This version is heavily optimized for speed on modern GPUs and produces a detailed JSON summary.

Author: Expert LLM Engineer
Usage: python quantum_transformer.py
"""

import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

# =====================================================================================
# CONFIGURATION (OPTIMIZED FOR SPEED)
# =====================================================================================

class Config:
    """Configuration class for the Quantum Transformer"""
    DATA_FILE = "/kaggle/input/quantum12/training_sequences.txt"
    SEQUENCE_LENGTH = 128
    VOCAB_SIZE = 3
    D_MODEL = 192
    N_HEADS = 6
    N_LAYERS = 4
    D_FF = 768
    DROPOUT = 0.1
    BATCH_SIZE = 256
    LEARNING_RATE = 5e-4
    EPOCHS = 20
    WEIGHT_DECAY = 0.01
    LOG_INTERVAL = 100
    EVAL_INTERVAL = 2
    SAMPLE_INTERVAL = 2
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    DPI = 300

    def __post_init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

config = Config()

# =====================================================================================
# TOKENIZER
# =====================================================================================

class BitTokenizer:
    """Simple tokenizer for binary sequences"""
    def __init__(self):
        self.token_to_id = {'0': 0, '1': 1, '<PAD>': 2}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id[char] for char in text if char in self.token_to_id]

    def decode(self, token_ids: List[int]) -> str:
        return ''.join([self.id_to_token[id] for id in token_ids if id in self.id_to_token])

# =====================================================================================
# DATASET (OPTIMIZED)
# =====================================================================================

class QuantumSequenceDataset(Dataset):
    """Dataset for quantum bit sequences (Optimized Version)"""
    def __init__(self, file_path: str, tokenizer: BitTokenizer, sequence_length: int):
        self.tokenizer = tokenizer
        self.sequences = []
        print(f"Loading data from {file_path}...")
        self._load_data(file_path, sequence_length)
        print(f"Loaded {len(self.sequences)} training examples of length {sequence_length}")

    def _load_data(self, file_path: str, sequence_length: int):
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Generating synthetic data...")
            self._generate_synthetic_data(file_path, num_sequences=10000)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == sequence_length:
                    self.sequences.append(self.tokenizer.encode(line))

    def _generate_synthetic_data(self, file_path: str, num_sequences: int):
        print("Generating synthetic quantum sequences...")
        with open(file_path, 'w') as f:
            for _ in range(num_sequences):
                sequence = [random.choice(['0', '1']) for _ in range(config.SEQUENCE_LENGTH)]
                f.write(''.join(sequence) + '\n')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        input_seq = full_seq[:-1]
        target_seq = full_seq[1:]
        return input_seq, target_seq

# =====================================================================================
# MODEL ARCHITECTURE (With fix for torch.compile and float16)
# =====================================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.w_q, self.w_k, self.w_v = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
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
        self.linear1, self.linear2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)
        self.dropout, self.activation = nn.Dropout(dropout), nn.GELU()
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention, self.feed_forward = MultiHeadAttention(d_model, n_heads, dropout), FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.dropout = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_output, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x, attn_weights

class QuantumTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model, self.max_seq_len = d_model, max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

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
# TRAINING AND VISUALIZATION UTILITIES
# =====================================================================================

class TrainingLogger:
    def __init__(self):
        self.train_losses, self.val_losses, self.learning_rates, self.epochs = [], [], [], []
        self.step_losses, self.step_numbers = [], []
    def log_step(self, step: int, loss: float, lr: float):
        self.step_numbers.append(step); self.step_losses.append(loss); self.learning_rates.append(lr)
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None):
        self.epochs.append(epoch); self.train_losses.append(train_loss)
        if val_loss is not None: self.val_losses.append(val_loss)
    def plot_training_curves(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10)); fig.suptitle('Training Progress Dashboard', fontsize=16, fontweight='bold')
        axes[0, 0].plot(self.epochs, self.train_losses, label='Training Loss'); axes[0, 0].plot(self.epochs, self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training & Validation Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        if self.step_losses: axes[0, 1].plot(self.step_numbers, self.step_losses, alpha=0.7); axes[0, 1].set_title('Step-wise Training Loss'); axes[0, 1].grid(True, alpha=0.3)
        if self.learning_rates: axes[1, 0].plot(self.step_numbers, self.learning_rates); axes[1, 0].set_title('Learning Rate Schedule'); axes[1, 0].set_yscale('log'); axes[1, 0].grid(True, alpha=0.3)
        if len(self.step_losses)>10: smoothed = np.convolve(self.step_losses, np.ones(10)/10, mode='valid'); axes[1, 1].plot(self.step_numbers[9:], smoothed); axes[1, 1].set_title('Smoothed Training Loss'); axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout();
        if save_path: plt.savefig(save_path, dpi=config.DPI);
        plt.show()

def calculate_perplexity(loss: float) -> float: return math.exp(loss)

def analyze_sequence_patterns(sequences: List[str]) -> Dict:
    analysis = {'bit_distribution': Counter(), 'bigram_distribution': Counter(), 'trigram_distribution': Counter(), 'entropy': [], 'run_lengths': {'0': [], '1': []}}
    for seq in sequences:
        analysis['bit_distribution'].update(seq)
        for i in range(len(seq) - 1): analysis['bigram_distribution'][seq[i:i+2]] += 1
        for i in range(len(seq) - 2): analysis['trigram_distribution'][seq[i:i+3]] += 1
        bit_counts = Counter(seq); total = len(seq)
        entropy = -sum((c/total) * math.log2(c/total) for c in bit_counts.values()) if total > 0 else 0
        analysis['entropy'].append(entropy)
        if not seq: continue
        current_bit, run_length = seq[0], 1
        for bit in seq[1:]:
            if bit == current_bit: run_length += 1
            else: analysis['run_lengths'][current_bit].append(run_length); current_bit, run_length = bit, 1
        analysis['run_lengths'][current_bit].append(run_length)
    return analysis

# =====================================================================================
# TRAINING LOOP (OPTIMIZED AND CORRECTED)
# =====================================================================================

def train_model(train_dataset: Dataset, val_dataset: Dataset, tokenizer: BitTokenizer):
    """Main training function (OPTIMIZED)"""
    print("🚀 Initializing OPTIMIZED Quantum Transformer Training Pipeline")
    print("=" * 60)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    model = QuantumTransformer(
        vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL, n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS, d_ff=config.D_FF, max_seq_len=config.SEQUENCE_LENGTH,
        dropout=config.DROPOUT
    ).to(device)

    print("Compiling model... (this may take a minute on first run)")
    model = torch.compile(model)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    logger = TrainingLogger()

    print("\n🎯 Starting Training"); print("=" * 60)
    best_val_loss, step = float('inf'), 0
    for epoch in range(config.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False)
        epoch_train_loss = 0.0
        for input_ids, target_ids in pbar:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_train_loss += loss.item(); step += 1
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})
            if step % config.LOG_INTERVAL == 0: logger.log_step(step, loss.item(), current_lr)

        avg_train_loss = epoch_train_loss / len(train_loader)
        
        if (epoch + 1) % config.EVAL_INTERVAL == 0:
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for input_ids, target_ids in val_loader:
                    input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                        logits, _ = model(input_ids)
                        loss = criterion(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.log_epoch(epoch, avg_train_loss, avg_val_loss)
            print(f"\n📊 Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {calculate_perplexity(avg_val_loss):.2f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                checkpoint = {'model_state_dict': uncompiled_model.state_dict(), 'config': config.__dict__}
                torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, 'best_model.pt'))
                print(f"   ✅ New best model saved! (Val Loss: {avg_val_loss:.4f})")

        if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
            print(f"\n🎲 Generating samples at epoch {epoch+1}")
            uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            generate_and_analyze_samples(uncompiled_model, tokenizer, device, epoch)
    
    print("\n✅ Training completed!")
    return model, logger

# =====================================================================================
# GENERATION, ANALYSIS & EVALUATION
# =====================================================================================

def generate_samples(model: nn.Module, tokenizer: BitTokenizer, device: torch.device,
                    num_samples: int = 5, max_length: int = 64, temperature: float = 1.0) -> List[str]:
    model.eval()
    samples = []
    max_context = model.max_seq_len - 1
    with torch.no_grad():
        for _ in range(num_samples):
            current_sequence = [random.choice([0, 1])]
            for _ in range(max_length - 1):
                context = current_sequence[-max_context:]
                input_tensor = torch.tensor([context], device=device)
                logits, _ = model(input_tensor)
                logits = logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token in [0, 1]: current_sequence.append(next_token)
                else: break
            samples.append(tokenizer.decode(current_sequence))
    return samples

def generate_and_analyze_samples(model: nn.Module, tokenizer: BitTokenizer, device: torch.device, epoch: int):
    samples = generate_samples(model, tokenizer, device, num_samples=10, max_length=64)
    print("   Sample sequences:"); [print(f"   {i+1:2d}: {s}") for i, s in enumerate(samples[:5])]
    analysis = analyze_sequence_patterns(samples)
    print(f"   Average entropy: {np.mean(analysis['entropy']):.3f} | Bit dist: {dict(analysis['bit_distribution'])}")

def final_evaluation(model: nn.Module, tokenizer: BitTokenizer, device: torch.device, val_dataset: Dataset, num_samples: int = 1000) -> Dict:
    """Performs a comprehensive final evaluation of the model against the validation set."""
    print("\n🔬 Performing Final Comprehensive Evaluation..."); print("=" * 60)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        eval_model = QuantumTransformer(
            vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL, n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS, d_ff=config.D_FF, max_seq_len=config.SEQUENCE_LENGTH, dropout=config.DROPOUT
        ).to(device)
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model loaded successfully.")
    else:
        print("Warning: No best model checkpoint found. Evaluating with the final model state.")
        eval_model = model
    eval_model.eval()
    
    print(f"Generating {num_samples} sequences for analysis...")
    generated_samples = generate_samples(eval_model, tokenizer, device, num_samples=num_samples, max_length=config.SEQUENCE_LENGTH - 1)
    original_samples = [tokenizer.decode(val_dataset[i][1].tolist()) for i in range(min(num_samples, len(val_dataset)))]
    
    print("Analyzing patterns in original and generated data...")
    original_analysis = analyze_sequence_patterns(original_samples)
    generated_analysis = analyze_sequence_patterns(generated_samples)
    
    summary = {
        'evaluation_summary': {'model_parameters': eval_model.count_parameters(), 'num_generated_samples': len(generated_samples), 'num_original_samples': len(original_samples)},
        'entropy_comparison': {'original_mean': np.mean(original_analysis['entropy']), 'original_std': np.std(original_analysis['entropy']), 'generated_mean': np.mean(generated_analysis['entropy']), 'generated_std': np.std(generated_analysis['entropy']), 'mean_difference': abs(np.mean(original_analysis['entropy']) - np.mean(generated_analysis['entropy']))},
        'bit_distribution': {'original': dict(original_analysis['bit_distribution']), 'generated': dict(generated_analysis['bit_distribution'])},
        'run_length_comparison_mean': {'original_0_run': np.mean(original_analysis['run_lengths']['0']) if original_analysis['run_lengths']['0'] else 0, 'generated_0_run': np.mean(generated_analysis['run_lengths']['0']) if generated_analysis['run_lengths']['0'] else 0, 'original_1_run': np.mean(original_analysis['run_lengths']['1']) if original_analysis['run_lengths']['1'] else 0, 'generated_1_run': np.mean(generated_analysis['run_lengths']['1']) if generated_analysis['run_lengths']['1'] else 0},
        'top_10_patterns': {'bigrams': {'original': original_analysis['bigram_distribution'].most_common(10), 'generated': generated_analysis['bigram_distribution'].most_common(10)}, 'trigrams': {'original': original_analysis['trigram_distribution'].most_common(10), 'generated': generated_analysis['trigram_distribution'].most_common(10)}}
    }
    return summary

# =====================================================================================
# VISUALIZATION UTILITIES
# =====================================================================================

def create_comprehensive_dashboard(logger: TrainingLogger, model: nn.Module, tokenizer: BitTokenizer, device: torch.device):
    fig = plt.figure(figsize=(20, 12)); fig.suptitle('Quantum Transformer - Complete Training Dashboard', fontsize=20, fontweight='bold'); gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2]); ax4 = fig.add_subplot(gs[0, 3])
    ax1.plot(logger.epochs, logger.train_losses, label='Training', lw=2); ax1.plot(logger.epochs, logger.val_losses, label='Validation', lw=2); ax1.set_title('Training Progress', fontweight='bold'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    if logger.learning_rates: ax2.plot(logger.step_numbers, logger.learning_rates, lw=2); ax2.set_title('Learning Rate Schedule', fontweight='bold'); ax2.set_xlabel('Step'); ax2.set_ylabel('Learning Rate'); ax2.set_yscale('log'); ax2.grid(True, alpha=0.3)
    if logger.val_losses: perplexities = [calculate_perplexity(loss) for loss in logger.val_losses]; ax3.plot(logger.epochs, perplexities, lw=2); ax3.set_title('Validation Perplexity', fontweight='bold'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('Perplexity'); ax3.grid(True, alpha=0.3)
    model_info = {'Parameters': f"{model.count_parameters():,}", 'Layers': config.N_LAYERS, 'Heads': config.N_HEADS, 'D_model': config.D_MODEL}; ax4.barh(np.arange(len(model_info)), [1]*len(model_info), color='lightblue', alpha=0.7); ax4.set_yticks(np.arange(len(model_info))); ax4.set_yticklabels([f"{k}: {v}" for k,v in model_info.items()]); ax4.set_title('Model Architecture', fontweight='bold'); ax4.set_xlim(0, 1); ax4.set_xticks([])
    ax5 = fig.add_subplot(gs[1, :2]); samples = generate_samples(model, tokenizer, device, num_samples=100, max_length=64); analysis = analyze_sequence_patterns(samples); bit_counts = [analysis['bit_distribution'].get('0', 0), analysis['bit_distribution'].get('1', 0)]; ax5.pie(bit_counts, labels=['0', '1'], autopct='%1.1f%%', startangle=90); ax5.set_title('Generated Bit Distribution', fontweight='bold')
    ax6 = fig.add_subplot(gs[1, 2]); ax6.hist(analysis['entropy'], bins=20, alpha=0.7, color='purple', edgecolor='black'); ax6.set_title('Entropy Distribution', fontweight='bold'); ax6.set_xlabel('Entropy'); ax6.grid(True, alpha=0.3)
    ax7 = fig.add_subplot(gs[1, 3]); all_runs_0, all_runs_1 = analysis['run_lengths']['0'], analysis['run_lengths']['1']
    if all_runs_0 and all_runs_1: max_run = max(max(all_runs_0, default=1), max(all_runs_1, default=1)); ax7.hist([all_runs_0, all_runs_1], bins=range(1, max_run + 2), alpha=0.7, label=['0-runs', '1-runs']); ax7.set_title('Run Length Distribution', fontweight='bold'); ax7.set_xlabel('Run Length'); ax7.legend(); ax7.grid(True, alpha=0.3)
    ax8 = fig.add_subplot(gs[2, :]); sample_text = "Sample Generated Sequences:\n\n" + "\n".join([f"{i+1:2d}: {s}" for i, s in enumerate(samples[:10])]); ax8.text(0.05, 0.95, sample_text, transform=ax8.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5)); ax8.axis('off'); ax8.set_title('Generated Sample Sequences', fontweight='bold', pad=20)
    plt.savefig(os.path.join(config.RESULTS_DIR, 'comprehensive_dashboard.png'), dpi=config.DPI, bbox_inches='tight'); plt.show()

def analyze_attention_patterns_comprehensive(model: nn.Module, tokenizer: BitTokenizer, device: torch.device, sample_sequence: str):
    model.eval(); tokens = tokenizer.encode(sample_sequence); input_tensor = torch.tensor([tokens], device=device)
    with torch.no_grad(): _, attention_weights = model(input_tensor)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12)); fig.suptitle('Attention Pattern Analysis', fontsize=16, fontweight='bold')
    num_layers = len(attention_weights)
    for i, (layer_idx, head_idx) in enumerate([(0, 0), (num_layers//2, 0), (num_layers-1, 0), (0, config.N_HEADS//2), (num_layers//2, config.N_HEADS-1), (num_layers-1, config.N_HEADS-1)]):
        row, col = i // 3, i % 3
        if layer_idx < num_layers and head_idx < config.N_HEADS:
            attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
            im = axes[row, col].imshow(attn, cmap='Blues', aspect='auto'); axes[row, col].set_title(f'Layer {layer_idx+1}, Head {head_idx+1}'); plt.colorbar(im, ax=axes[row, col])
    plt.tight_layout(); plt.savefig(os.path.join(config.RESULTS_DIR, 'attention_patterns.png'), dpi=config.DPI, bbox_inches='tight'); plt.show()

# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution function"""
    print("🌟 QUANTUM BIT SEQUENCE TRANSFORMER (OPTIMIZED) 🌟"); print("=" * 60)
    print("\n⚙️ Configuration:")
    for key, value in config.__dict__.items():
        if not key.startswith('__') and not key.startswith('_'): print(f"   {key}: {value}")
    
    tokenizer = BitTokenizer()
    dataset = QuantumSequenceDataset(config.DATA_FILE, tokenizer, config.SEQUENCE_LENGTH)
    train_size = int(0.9 * len(dataset)); val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    model, logger = train_model(train_dataset, val_dataset, tokenizer)
    
    uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n📈 Generating final visualizations...")
    logger.plot_training_curves(os.path.join(config.RESULTS_DIR, 'training_curves.png'))
    create_comprehensive_dashboard(logger, uncompiled_model, tokenizer, device)
    
    print("\n🔍 Analyzing attention patterns...")
    sample_seq = "10101101001110010101101001110010"[:config.SEQUENCE_LENGTH-1]
    analyze_attention_patterns_comprehensive(uncompiled_model, tokenizer, device, sample_seq)
    
    final_summary_data = final_evaluation(uncompiled_model, tokenizer, device, val_dataset)
    
    print("\n" + "="*25 + " FINAL SUMMARY " + "="*25)
    print(f"Model Parameters: {final_summary_data['evaluation_summary']['model_parameters']:,}"); print("-" * 60)
    print("ENTROPY (Higher is more random, 1.0 is max for binary)")
    print(f"  - Original Data:  Mean={final_summary_data['entropy_comparison']['original_mean']:.4f}, StdDev={final_summary_data['entropy_comparison']['original_std']:.4f}")
    print(f"  - Generated Data: Mean={final_summary_data['entropy_comparison']['generated_mean']:.4f}, StdDev={final_summary_data['entropy_comparison']['generated_std']:.4f}")
    print(f"  - Model learned entropy with a difference of: {final_summary_data['entropy_comparison']['mean_difference']:.4f}"); print("-" * 60)
    orig_bits = final_summary_data['bit_distribution']['original']; gen_bits = final_summary_data['bit_distribution']['generated']
    print("BIT DISTRIBUTION (0s vs 1s)")
    print(f"  - Original Data:  0: {orig_bits.get('0', 0)}, 1: {orig_bits.get('1', 0)}")
    print(f"  - Generated Data: 0: {gen_bits.get('0', 0)}, 1: {gen_bits.get('1', 0)}"); print("-" * 60)
    runs = final_summary_data['run_length_comparison_mean']
    print("MEAN RUN LENGTH (Average length of consecutive identical bits)")
    print(f"  - '0' Runs: Original={runs['original_0_run']:.2f}, Generated={runs['generated_0_run']:.2f}")
    print(f"  - '1' Runs: Original={runs['original_1_run']:.2f}, Generated={runs['generated_1_run']:.2f}"); print("-" * 60)
    print("TOP 5 BIGRAMS (Two-bit patterns)")
    print(f"  - Original:  {final_summary_data['top_10_patterns']['bigrams']['original'][:5]}")
    print(f"  - Generated: {final_summary_data['top_10_patterns']['bigrams']['generated'][:5]}"); print("-" * 60)
    print("TOP 5 TRIGRAMS (Three-bit patterns)")
    print(f"  - Original:  {final_summary_data['top_10_patterns']['trigrams']['original'][:5]}")
    print(f"  - Generated: {final_summary_data['top_10_patterns']['trigrams']['generated'][:5]}"); print("=" * 65)

    json_path = os.path.join(config.RESULTS_DIR, 'final_summary.json')
    print(f"\n💾 Saving detailed analysis to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(final_summary_data, f, indent=2)

    print("\n✨ All tasks complete!")
    return model, tokenizer, logger

if __name__ == "__main__":
    config.__post_init__()
    main()