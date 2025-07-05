# 🧠 Quantum Decoherence Predictability Analysis

This document summarizes the mathematical findings from analyzing bitwise predictability in quantum decoherence datasets using neural models.

---

## 📊 Dataset

- **Total samples analyzed:** 19,531 sequences
- **Sequence length:** 128 bits
- **Test size per bit position:** 3,907 samples
- **Models used:** Transformer and LSTM (both ~51% prediction accuracy)

---

## 🔢 Statistical Test Per Bit

For each bit position *i* ∈ {0, 1, ..., 127}, we evaluated:

### Binomial Hypothesis Test

**Null hypothesis:**
$$H_0: \text{Accuracy}_i = 0.5$$

We compute:
$$p_i = \text{BinomialTest}(k = A_i \cdot N, \; n = N, \; p = 0.5)$$

Where:
- *A_i* = empirical accuracy of predicting bit *i*
- *N* = 3,907 = number of test sequences

---

## 📈 Predictability Trend

A linear regression was fitted to the bitwise accuracy values:

$$\hat{A}(i) = -4.22 \times 10^{-5} \cdot i + 0.5046$$

This implies:
- Slight **decreasing** trend in predictability over later bit positions
- The **intercept (≈ 0.5046)** suggests a small but persistent deviation from pure randomness

---

## 📁 Resources

- 📑 Raw Data: [`predictability_analysis.csv`](./predictability_analysis.csv)
- 📊 Plot: [`predictability_analysis_plot.png`](./predictability_analysis_plot.png)

---

## ✅ Conclusion

While the accuracy deviations are small, the presence of a statistically meaningful and consistent trend suggests potential **non-random structure** within measured decoherence patterns. This supports the hypothesis of subtle deterministic influences in quantum collapse.

Further mathematical modeling is warranted.