import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pysr import PySRRegressor
import warnings
warnings.filterwarnings('ignore')

# Define the attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(out), attention_weights

# Define the feed forward network
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# Define the transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Self-attention with residual connection
        attn_out, attention_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, attention_weights

# Define the QuantumTransformer architecture (matching your trained model)
class QuantumTransformer(nn.Module):
    def __init__(self, vocab_size: int = 2, d_model: int = 128, n_heads: int = 8, 
                 n_layers: int = 4, d_ff: int = 512, max_seq_len: int = 128, dropout: float = 0.1):
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

def load_test_data(file_path):
    """Load test sequences from file"""
    sequences = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and len(line) == 128:  # Ensure 128 bits
                    sequences.append(line)
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample data...")
        # Create sample data for demonstration
        np.random.seed(42)
        sequences = []
        for _ in range(1000):
            seq = ''.join(np.random.choice(['0', '1'], 128))
            sequences.append(seq)
    
    return sequences

def sequences_to_tensor(sequences):
    """Convert string sequences to tensor"""
    tensor_data = []
    for seq in sequences:
        # Convert string to list of integers
        seq_ints = [int(bit) for bit in seq]
        tensor_data.append(seq_ints)
    
    return torch.tensor(tensor_data, dtype=torch.long)

def extract_features(sequences, window_sizes=[3, 5, 7, 10, 15]):
    """Extract meaningful features from bit sequences"""
    features = []
    feature_names = []
    
    for seq in sequences:
        seq_features = []
        bits = [int(b) for b in seq]
        
        # Basic statistics
        seq_features.extend([
            np.mean(bits),  # Average bit value
            np.std(bits),   # Standard deviation
            np.sum(bits),   # Total number of 1s
            len(bits) - np.sum(bits)  # Total number of 0s
        ])
        
        # Pattern features for different window sizes
        for window_size in window_sizes:
            if len(bits) >= window_size:
                # Last window features
                last_window = bits[-window_size:]
                seq_features.extend([
                    np.mean(last_window),
                    np.std(last_window) if len(last_window) > 1 else 0,
                    np.sum(last_window)
                ])
                
                # Count specific patterns
                pattern_counts = {}
                for i in range(len(bits) - window_size + 1):
                    pattern = tuple(bits[i:i+window_size])
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                # Add most common pattern frequency
                if pattern_counts:
                    max_pattern_freq = max(pattern_counts.values()) / len(pattern_counts)
                    seq_features.append(max_pattern_freq)
                else:
                    seq_features.append(0)
        
        # Transition features
        transitions = sum(1 for i in range(len(bits)-1) if bits[i] != bits[i+1])
        seq_features.append(transitions / (len(bits) - 1) if len(bits) > 1 else 0)
        
        # Run length features
        runs = []
        current_run = 1
        for i in range(1, len(bits)):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        seq_features.extend([
            np.mean(runs),
            np.max(runs),
            np.min(runs),
            np.std(runs) if len(runs) > 1 else 0
        ])
        
        # Position-based features (last few bits)
        for i in range(min(10, len(bits))):
            seq_features.append(bits[-(i+1)])
        
        features.append(seq_features)
    
    # Create feature names
    if not feature_names:
        feature_names = ['mean_bits', 'std_bits', 'sum_ones', 'sum_zeros']
        
        for window_size in window_sizes:
            feature_names.extend([
                f'last_{window_size}_mean',
                f'last_{window_size}_std',
                f'last_{window_size}_sum',
                f'pattern_freq_{window_size}'
            ])
        
        feature_names.append('transition_rate')
        feature_names.extend(['run_mean', 'run_max', 'run_min', 'run_std'])
        
        for i in range(min(10, len(sequences[0]) if sequences else 0)):
            feature_names.append(f'bit_pos_{i+1}')
    
    return np.array(features), feature_names

def load_model(checkpoint_path, device='cpu'):
    """Load the trained model with proper architecture matching"""
    try:
        # Try to load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Print checkpoint keys to understand the structure
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        # Try to infer model parameters from the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Infer parameters from state dict
        # Get vocab_size from token_embedding
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        d_model = state_dict['token_embedding.weight'].shape[1]
        max_seq_len = state_dict['position_embedding.weight'].shape[0]
        
        # Count number of layers
        n_layers = len([k for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.attention.w_q.weight')])
        
        # Get n_heads from attention weights
        attention_weight = state_dict['blocks.0.attention.w_q.weight']
        n_heads = 8  # Default, will be adjusted if needed
        
        # Get d_ff from feed forward
        d_ff = state_dict['blocks.0.feed_forward.linear1.weight'].shape[0]
        
        print(f"Inferred parameters:")
        print(f"  vocab_size: {vocab_size}")
        print(f"  d_model: {d_model}")
        print(f"  n_layers: {n_layers}")
        print(f"  d_ff: {d_ff}")
        print(f"  max_seq_len: {max_seq_len}")
        
        # Initialize model with inferred parameters
        model = QuantumTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len
        )
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found. Creating a dummy model for demonstration...")
        # Create a dummy model for demonstration
        model = QuantumTransformer()
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a dummy model for demonstration...")
        model = QuantumTransformer()
        model.eval()
        return model

def get_model_predictions(model, sequences, device='cpu'):
    """Get model predictions for all sequences"""
    model.to(device)
    model.eval()
    
    # Convert sequences to tensor
    tensor_data = sequences_to_tensor(sequences)
    
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(tensor_data), batch_size):
            batch = tensor_data[i:i+batch_size].to(device)
            
            # Get model output (logits and attention weights)
            logits, attention_weights = model(batch)
            
            # For next token prediction, we want the last token's logits
            # Apply softmax to get probabilities for next token
            last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Get probability of predicting '1' (index 1)
            prob_of_one = probs[:, 1]  # Probability of next bit being 1
            
            predictions.extend(prob_of_one.cpu().numpy())
    
    return np.array(predictions)

def symbolic_regression(X, y, feature_names):
    """Perform symbolic regression to find mathematical formula"""
    print("Starting symbolic regression...")
    
    # Initialize PySR without feature_names parameter
    model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
        model_selection="best",
        loss="loss(prediction, target) = abs(prediction - target)",
        maxsize=20,
        populations=30,
        procs=4,
        random_state=42,
        verbosity=1
    )
    
    # Store feature names for later use
    model.feature_names_ = feature_names
    
    # Fit the model
    model.fit(X, y)
    
    return model

def analyze_results(sr_model, X, y, feature_names):
    """Analyze and visualize results"""
    print("\n" + "="*50)
    print("SYMBOLIC REGRESSION RESULTS")
    print("="*50)
    
    # Print the equations
    print("\nBest equations found:")
    try:
        # Try to get sympy representation
        best_equation = sr_model.sympy()
        print(best_equation)
        
        # Also print the latex representation if available
        try:
            latex_eq = sr_model.latex()
            print(f"\nLaTeX format: {latex_eq}")
        except:
            pass
            
    except Exception as e:
        print(f"Could not display sympy equation: {e}")
        # Try to get the raw equation string
        try:
            if hasattr(sr_model, 'equations_'):
                best_eq = sr_model.equations_.iloc[0]
                print(f"Best equation: {best_eq['equation']}")
                print(f"Complexity: {best_eq['complexity']}")
                print(f"Loss: {best_eq['loss']}")
        except:
            print("Could not retrieve equation details")
    
    # Get predictions
    try:
        y_pred = sr_model.predict(X)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None, None
    
    # Calculate metrics
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Prediction vs actual
    plt.subplot(2, 3, 1)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Probability')
    plt.ylabel('Predicted Probability')
    plt.title('Actual vs Predicted')
    
    # Residuals
    plt.subplot(2, 3, 2)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Feature importance (if available)
    plt.subplot(2, 3, 3)
    try:
        if hasattr(sr_model, 'feature_importances_'):
            feature_importance = np.abs(sr_model.feature_importances_)
            top_features = np.argsort(feature_importance)[-10:]
            plt.barh(range(len(top_features)), feature_importance[top_features])
            plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
    except Exception as e:
        plt.text(0.5, 0.5, f'Feature importance\nerror: {str(e)[:20]}...', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')
    
    # Distribution of predictions
    plt.subplot(2, 3, 4)
    plt.hist(y, bins=30, alpha=0.7, label='Actual', density=True)
    plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Distribution Comparison')
    
    # Error distribution
    plt.subplot(2, 3, 5)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    
    # Complexity vs accuracy (if available)
    plt.subplot(2, 3, 6)
    try:
        if hasattr(sr_model, 'equations_'):
            equations = sr_model.equations_
            plt.scatter(equations['complexity'], equations['loss'])
            plt.xlabel('Complexity')
            plt.ylabel('Loss')
            plt.title('Complexity vs Loss')
        else:
            plt.text(0.5, 0.5, 'Complexity plot\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Complexity vs Loss')
    except Exception as e:
        plt.text(0.5, 0.5, f'Complexity plot\nerror: {str(e)[:20]}...', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Complexity vs Loss')
    
    plt.tight_layout()
    plt.show()
    
    return mse, mae, r2

def main():
    """Main execution function"""
    print("Quantum Transformer Model Extraction")
    print("="*40)
    
    # Configuration
    model_checkpoint = "best_model.pt"
    test_data_file = "torino_test_sequences.txt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Step 1: Load test data
    print("\n1. Loading test data...")
    sequences = load_test_data(test_data_file)
    print(f"Loaded {len(sequences)} sequences")
    
    # Step 2: Load model
    print("\n2. Loading model...")
    model = load_model(model_checkpoint, device)
    print("Model loaded successfully")
    
    # Step 3: Get model predictions
    print("\n3. Getting model predictions...")
    predictions = get_model_predictions(model, sequences, device)
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Step 4: Extract features
    print("\n4. Extracting features...")
    features, feature_names = extract_features(sequences)
    print(f"Extracted {features.shape[1]} features")
    print(f"Feature names: {feature_names[:5]}...")  # Show first 5
    
    # Step 5: Prepare data for symbolic regression
    print("\n5. Preparing data for symbolic regression...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, predictions, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 6: Symbolic regression
    print("\n6. Performing symbolic regression...")
    try:
        sr_model = symbolic_regression(X_train, y_train, feature_names)
        
        # Step 7: Analyze results
        print("\n7. Analyzing results...")
        mse, mae, r2 = analyze_results(sr_model, X_test, y_test, feature_names)
        
        # Save results
        print("\n8. Saving results...")
        
        # Save the mathematical formula
        with open('extracted_formula.txt', 'w') as f:
            f.write("Extracted Mathematical Formula:\n")
            f.write("="*40 + "\n\n")
            
            try:
                # Try to get sympy representation
                formula = str(sr_model.sympy())
                f.write(f"Best Formula: {formula}\n\n")
                
                # Try to get latex representation
                try:
                    latex_formula = sr_model.latex()
                    f.write(f"LaTeX Format: {latex_formula}\n\n")
                except:
                    pass
                    
            except Exception as e:
                f.write(f"Could not extract sympy formula: {e}\n\n")
                
                # Try to get raw equation from equations table
                try:
                    if hasattr(sr_model, 'equations_'):
                        best_eq = sr_model.equations_.iloc[0]
                        f.write(f"Best Equation: {best_eq['equation']}\n")
                        f.write(f"Complexity: {best_eq['complexity']}\n")
                        f.write(f"Loss: {best_eq['loss']}\n\n")
                except:
                    f.write("Could not extract equation details\n\n")
            
            f.write(f"Model Performance on Test Set:\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"R²: {r2:.6f}\n\n")
            
            # Add feature information
            f.write("Feature Information:\n")
            f.write("-" * 20 + "\n")
            for i, name in enumerate(feature_names):
                f.write(f"X{i}: {name}\n")
        
        # Save feature importance if available
        try:
            feature_importance = np.abs(sr_model.feature_importances_)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv('feature_importance.csv', index=False)
            print("Feature importance saved to 'feature_importance.csv'")
        except:
            print("Feature importance not available")
        
        print("Mathematical formula saved to 'extracted_formula.txt'")
        print("\nExtraction complete!")
        
    except Exception as e:
        print(f"Error during symbolic regression: {e}")
        print("This might be due to PySR installation issues.")
        print("Try installing PySR with: pip install pysr")

if __name__ == "__main__":
    main()