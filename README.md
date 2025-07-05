

# 🧪 Quantum Decoherence: Detecting Patterns in Randomness

**By Ibrahim Khamis ElDosoky**
*Research into the statistical behavior of quantum decoherence using real quantum computers and deep learning.*

---

## 🔍 Overview

This project presents compelling evidence that **quantum decoherence is not perfectly random**. Using large-scale real measurements from IBM quantum devices, deep learning models (LSTM and Transformer) were trained to detect patterns in collapse outcomes. Results consistently outperformed chance, challenging the foundation of quantum randomness.

---

## 🧠 Key Insights

* **500k-shot datasets** collected from `ibm_brisbane` and `ibm_torino`.

* **LSTM and Transformer models** achieved \~51% prediction accuracy on sequences labeled random.

* **Symbolic regression** extracted a closed-form equation from a trained model:

  ```
  next_bit_prob ≈ last_bit × 0.00104 + 0.49641
  ```

* **Entropy ≈ 0.999+** across all bits — suspiciously perfect.

* **Bitwise predictability** shows positional bias in decoherence.

* **Cross-device testing (brisbane vs torino)** showed consistent model performance.

---

## 📊 Visuals & Results

* 📈 `bitwise_entropy.png`: Entropy per bit across 128-bit sequences
* 📊 `predictability_vs_bit_position.png`: Predictability by bit position
* 🧠 `attention_patterns.png`: Attention map from Transformer model
* 📑 `comprehensive_dashboard.png`: Combined insight dashboard
* 📜 `Quantum_Decoherence_Math_Summary.md`: Theoretical implications and math

---

## 🧪 Project Files

| File                            | Purpose                                       |
| ------------------------------- | --------------------------------------------- |
| `training_sequences.txt`        | 500k quantum measurements from `ibm_brisbane` |
| `torino_test_sequences.txt`     | Validation from `ibm_torino`                  |
| `best_model.pt`                 | Trained transformer checkpoint                |
| `train.py`, `test.py`           | Model training & evaluation                   |
| `quantum_model_extraction.py`   | Symbolic regression on model behavior         |
| `first_collaps.py`              | First-bit entropy and chi-square test         |
| `predictability_vs_position.py` | Analyze pattern strengths by bit              |
| `README.md`                     | Project documentation                         |

---

## 📂 Directory Tree

```
.
├── training_sequences.txt
├── torino_test_sequences.txt
├── simulated_sequences.txt
├── best_model.pt
├── train.py / test.py
├── quantum_model_extraction.py
├── predictability_vs_bit_position.png
├── bitwise_entropy.png
├── attention_patterns.png
├── README.md
└── ... (see directory listing)
```

---

## 📌 Conclusions

This project provides:

* Statistical evidence of **non-random structure** in quantum measurements.
* A foundation for **new physical theories** involving structured decoherence.
* A reproducible method to evaluate real quantum randomness at scale.

---

## 🗞️ Contact & Media

I am seeking collaboration or media coverage to push this discovery further.

📧 **[ibrahimkhamiseldosoky@gmail.com](mailto:ibrahimkhamiseldosoky@gmail.com)**


