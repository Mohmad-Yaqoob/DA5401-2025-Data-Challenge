# 📘 Metric Learning Score Prediction — DA5401 End-Semester Challenge

## 🧩 Project Overview

Welcome to the **DA5401 End-Semester Data Challenge**.  
This project focuses on **metric learning**, a sub-area of machine learning centered on learning similarity functions between objects.

The challenge aims to evaluate the *fitness* between:

- **Metric Definition** (text embedding)
- **Prompt–Response Pair** (text embedding)

The goal is to predict a **score between 0 and 10** that represents how well the prompt–response pair aligns with a specific evaluation metric.

### 🧠 Why This Matters
In practical conversational AI evaluation:
- We need **high-quality test datasets**
- Each test prompt must match the **intended evaluation metric**
- Manual evaluation is expensive
- LLM-based automatic scoring introduces variability

This task allows us to build an automated scoring model that:
- Predicts similarity between two embeddings  
- Helps evaluate AI systems reliably  
- Improves dataset quality and testing coverage  

---

## 📂 Dataset

The dataset contains:

### ✔ Metric Name Embeddings (`metric_name_embeddings.npy`)
Vector representation (via Gemma Embedding Model) of each metric/submetric.

### ✔ Train Data (`train_data.json`)
Contains:
- **metric_name**
- **user_prompt**
- **response**
- **system_prompt**
- **score (0–10)**

### ✔ Test Embeddings  
Created by you using SentenceTransformer:
- `test_metric_embs.npy`
- `test_text_embs.npy`

These correspond to metric definitions and prompt-response pairs for test samples.

---

## 🔧 Feature Engineering

For each train/test item, we compute:

1. **Cosine similarity**  
2. **Absolute difference** (embedding-wise)  
3. **Element-wise product**  
4. **Concatenation** of embeddings  
5. Final feature vector size ≈ 3073  

This provides the model with both interaction features and raw representations.

---

## 🏗️ Model: ResNet-MLP

A custom deep MLP with:

- Input Projection  
- **8 Residual MLP Blocks**  
- Pre-LayerNorm  
- GELU Activation  
- Dropout  
- LayerNorm + Final Linear Head  

The architecture is highly stable for large dense vectors.

---

## 🎯 Loss Function

We use a combination:

```
Total Loss = MSE + λ * KL Histogram Loss
```

### 🔸 MSE Loss  
Ensures accuracy on individual samples.

### 🔸 KL Histogram Loss  
Ensures predicted score distribution matches target score PDF  
(prevents collapse, improves calibration).

---

## 🛠️ Training Strategy

### ✔ 5-Fold Cross-Validation  
Prevents overfitting, provides robust OOF predictions.

### ✔ SWA (Stochastic Weight Averaging)  
Smooths weights for better generalization.

### ✔ EMA (Exponential Moving Average)  
Creates an additional stabilized model.

### ✔ Cosine Annealing LR  
Gradually reduces learning rate.

### ✔ Early Stopping  
Triggers if validation RMSE stops improving.

---

## 📊 Post-Processing

### ✔ Linear Calibration  
Fits:
```
y = a * prediction + b
```

### ✔ Per-Fold Quantile Mapping  
For each fold:
- Align prediction distribution with target distribution  
- Stabilizes test predictions  

### ✔ Final Averaging + Clipping  
Mean of all folds  
Values clipped to `[0, 10]`.

---

## 📄 Final Submission

The generated file:

```
submission_resnetmlp_histKL_swa_ema_perfold.csv
```

Contains:

```
ID, score
1, 8.57
2, 6.44
...
```

---

## 📊 Visualizations

You produced two categories:

### **1. Dataset Visualization**
Saved in `figures_dataset/`:
- PCA & t-SNE of metric embeddings  
- Cosine similarity heatmap  
- Prompt/response length distributions  
- Score distributions  

### **2. Final Submission Visualization**
Saved in `figures_submission/`:
- Prediction histogram  
- KDE plot  
- Boxplot  
- Violin plot  
- CDF plot  
- Outlier detection  
- Score bucket heatmap  

---

## 📁 Folder Structure

```
project/
│
├── X_all.npy
├── y_all.npy
├── test_metric_embs.npy
├── test_text_embs.npy
│
├── figures_dataset/
├── figures_submission/
│
├── submission_resnetmlp_histKL_swa_ema_perfold.csv
│
└── README.md
```

---

## ▶️ How to Run (If Retrained)

```
python main_training_script.py
```

To visualize:

```
python visualize_submission.py
```

---

## 🌟 Conclusion

This project successfully implements a **high-performance metric learning pipeline** using advanced deep learning techniques such as:

- ResNet-style MLP  
- KL divergence distribution matching  
- SWA & EMA  
- K-fold CV  
- Quantile mapping  
- Calibration  

It produces highly reliable fitness predictions for conversational AI evaluation tasks.

---

## ✨ Author  
**Mohmad Yaqoob**  
M.Tech, IIT Madras  
DA5401 – End Semester Challenge, 2025

