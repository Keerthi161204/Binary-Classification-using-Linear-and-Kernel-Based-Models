# Experiment 4 – Binary Classification using Linear and Kernel-Based Models

This repository contains **Experiment 4** from the *Machine Learning Algorithms Laboratory*.  
The experiment focuses on implementing **Logistic Regression** and **Support Vector Machine (SVM)** classifiers to perform **binary email spam classification**, along with **hyperparameter tuning** and **cross-validation**.

---

## 📌 Experiment Details

- **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
- **Affiliation:** Anna University  
- **Degree & Branch:** B.E. Computer Science & Engineering  
- **Semester:** VI  
- **Subject Code & Name:** UCS2612 – Machine Learning Algorithms Laboratory  
- **Academic Year:** 2025–2026 (Even Semester)  
- **Batch:** 2023–2027  

---

## 🎯 Aim

To classify emails as **Spam** or **Ham** using:
- Logistic Regression
- Support Vector Machine (SVM)

and analyze the effect of **hyperparameter tuning** on classification performance.

---

## 📂 Dataset

- **Spambase Dataset**
- Consists of numerical features extracted from email content
- Binary target variable:
  - `0` → Ham (Non-spam)
  - `1` → Spam

---

## 🎯 Objectives

- Implement Logistic Regression and SVM classifiers  
- Tune hyperparameters using **Grid Search**  
- Compare **kernel behavior** in SVM  
- Evaluate models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score  
- Perform **5-fold Cross Validation**

---

## 🧰 Libraries Used

- **Pandas** – Data manipulation  
- **NumPy** – Numerical computation  
- **Matplotlib** – Visualization  
- **Seaborn** – Statistical visualization  
- **Scikit-learn** – Model building, preprocessing, evaluation  

---

## 🤖 Machine Learning Models Used

- **Logistic Regression**
- **Support Vector Machine (SVM)**
  - Linear kernel
  - Polynomial kernel
  - RBF kernel
  - Sigmoid kernel

---

## 🧪 Experiment Workflow

### 1️⃣ Data Loading
- Load Spambase dataset
- Separate features and target label

### 2️⃣ Data Preprocessing
- Feature scaling using `StandardScaler`

### 3️⃣ Train–Test Split
- 80% training data
- 20% testing data (stratified split)

### 4️⃣ Logistic Regression
- Baseline Logistic Regression model
- Hyperparameter tuning using GridSearchCV

### 5️⃣ Support Vector Machine
- Train SVM with different kernels
- Measure accuracy, F1-score, and training time
- Hyperparameter tuning using GridSearchCV

### 6️⃣ Cross Validation
- Perform 5-fold cross-validation
- Compare Logistic Regression and SVM

### 7️⃣ Model Evaluation & Visualization
- Classification reports
- Confusion matrices
- ROC curves
- Accuracy comparison plots

---

## 📊 Performance Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC  
- Training Time  

---

## ⚙️ Hyperparameter Tuning Results

### Table 1: Hyperparameter Tuning Summary

| Model | Search Method | Best Parameters | Best CV Accuracy |
|------|--------------|-----------------|------------------|
| Logistic Regression | GridSearch | C, penalty, solver | 0.9315 |
| SVM | GridSearch | Kernel, C, gamma | 0.9359 |

---

## 📈 Logistic Regression Performance

### Table 2: Logistic Regression Metrics

| Metric | Value |
|------|-------|
| Accuracy | 0.93 |
| Precision | 0.93 |
| Recall | 0.95 |
| F1 Score | 0.94 |

---

## 📊 SVM Kernel-wise Performance

### Table 3: SVM Kernel Comparison

| Kernel | Accuracy | F1 Score |
|------|----------|----------|
| Linear | 0.93 | 0.94 |
| Polynomial | 0.78 | 0.84 |
| RBF | 0.93 | 0.94 |
| Sigmoid | 0.88 | 0.90 |

---

## 🔁 K-Fold Cross Validation

### Table 4: 5-Fold Cross Validation Results

| Fold | Logistic Regression | SVM |
|-----|---------------------|-----|
| Fold 1 | 0.9186 | 0.9349 |
| Fold 2 | 0.9272 | 0.9337 |
| Fold 3 | 0.9293 | 0.9228 |
| Fold 4 | 0.9315 | 0.9359 |
| Fold 5 | 0.9315 | 0.9304 |
| **Average** | **0.9293** | **0.9337** |

---

## 📷 Output Visualizations

- Logistic Regression confusion matrix  
- Logistic Regression ROC curve  
- SVM confusion matrix  
- SVM ROC curve  
- SVM kernel-wise comparison plots  
- Cross-validation accuracy comparison bar chart  

---

## 🔍 Observations

- **Linear SVM** achieved the best overall performance
- Regularization improved Logistic Regression generalization
- SVM kernel choice significantly influenced performance
- Polynomial kernel performed poorly compared to Linear and RBF kernels

---

## 🧠 Learning Outcomes

From this experiment, students learned:

- How linear and kernel-based classifiers work
- Practical implementation of Logistic Regression and SVM
- Hyperparameter tuning using GridSearchCV
- Model evaluation using multiple metrics
- Interpretation of confusion matrices and ROC curves
- Importance of cross-validation in performance estimation
- Comparative analysis of classifiers

---

