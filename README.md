---

# Leakage-Aware Regression

A leakage-aware regression pipeline that demonstrates how to detect **target leakage**, remove leaking features, train a regression model, and evaluate performance using **Mean Squared Error (MSE)**.

This project uses a synthetic dataset to simulate a real-world scenario where a feature contains information derived from the target and must be removed before training.

---

## Highlights

* Generates a synthetic regression dataset (`x1`, `x2`, `target`)
* Injects a leakage feature (`leak_feature`)
* Detects leakage using correlation analysis
* Removes leaking feature(s) automatically
* Trains **Linear Regression**
* Saves predictions and evaluation metrics to disk

---

## Project Structure

```
leakage-aware-regression/
├── src/
│   └── main.py
├── outputs/
│   ├── predictions.csv
│   ├── mse.txt
│   └── removed_features.txt
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Clone the repository

```bash
git clone https://github.com/theshloksschauhan/leakage-aware-regression.git
cd leakage-aware-regression
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run

```bash
python src/main.py
```

---

## Output

After running, the following files are generated inside `outputs/`:

* `removed_features.txt` — features removed due to leakage detection
* `predictions.csv` — model predictions
* `mse.txt` — Mean Squared Error (MSE)

---

## How It Works (High-Level)

1. Generate regression data with valid features
2. Create a leakage feature derived from the target
3. Compute feature-target correlation
4. Remove features with extremely high correlation to target
5. Train Linear Regression on clean features
6. Evaluate using MSE and save outputs

---

## Why This Matters

Target leakage causes unrealistic model performance during training/testing and breaks real-world deployment.
This project demonstrates a simple, effective approach to maintain evaluation integrity by detecting and removing leakage features.

---

## Author

**Shlok Shoorveer Singh Chauhan**
GitHub: [https://github.com/theshloksschauhan](https://github.com/theshloksschauhan)

---
