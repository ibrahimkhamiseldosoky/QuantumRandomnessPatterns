
# Quantum Randomness Pattern Discovery

**By Ibrahim Khamis El Dosoky**

This project investigates the predictability of quantum decoherence patterns from real quantum computers, challenging the assumption that quantum measurements are truly random.

## 🔬 Summary

We used 500,000-shot measurements from IBM quantum devices to build a dataset of quantum bit collapses. Then we applied deep learning techniques (LSTM and Transformer models) to detect patterns in the data. The models consistently outperformed random baselines, suggesting **non-randomness** in the output distribution of supposedly "pure" quantum randomness.

## 📈 Key Findings

* **LSTM improved over random guessing by \~51% accuracy**, detecting subtle but consistent correlations.

* **Symbolic regression** discovered a closed-form approximation of the model’s learned logic:

  ```
  next_bit_probability = last_bit × 0.00104 + 0.49641
  ```

* **Entropy analysis** showed entropy values suspiciously close to perfect (1.0), far more than expected from natural quantum noise.

* **Bitwise predictability analysis** revealed that some bits are more predictable than others, indicating structured statistical leakage in the collapse process.

## 🧠 Implications

This result suggests that **quantum decoherence may not be entirely random**, which could have serious consequences for:

* **Quantum security and QRNGs** (Quantum Random Number Generators)
* **Foundations of quantum mechanics**
* **Theories of consciousness** (supporting ideas that consciousness may be tied to quantum collapse)

## 📦 Project Structure

```
├── best_model.pt                    # Trained transformer model
├── training_sequences.txt          # Raw training data
├── simulated_sequences.txt         # Classical simulation for comparison
├── torino_test_sequences.txt       # Dataset from second quantum device
├── visuals/                        # All analysis graphs and plots
├── final_summary.json              # All stats and metrics
├── test_results/                   # Evaluation reports
├── extracted_formula.txt           # Symbolic regression result
├── quantum_model_extraction.py     # Extracts interpretable formula from model
├── predictability_vs_bit_position.png # Predictability per bit plot
└── README.md                       # You are here
```

## 🔍 Reproducibility

This repo contains:

* Code to run deep learning analysis on quantum data
* Pretrained models
* Visualizations and statistical outputs
* Python scripts for entropy, predictability, and regression

All code can be run on a local machine with `PyTorch`, `scikit-learn`, `sympy`, and optionally `Julia` via `PySR` for symbolic regression.

## 🧪 Devices Used

* **ibm\_brisbane** – 500k real quantum shots
* **ibm\_torino** – 500k real quantum shots (validation)
* Classical simulation for baseline comparison

## 📢 Contact & Media

If you're a researcher, journalist, or quantum computing enthusiast interested in the implications of this work:

📧 **[ibrahimkhamiseldosoky@gmail.com](mailto:ibrahimkhamiseldosoky@gmail.com)**

