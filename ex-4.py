# ===============================
# EXPERIMENT 4 : LOGISTIC + SVM
# ===============================

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("../Dataset/4_spambase/spambase_csv.csv")

X = df.drop("class", axis=1)
y = df["class"]

# -------------------------------
# Preprocessing
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Logistic Regression Baseline
# -------------------------------
log_model = LogisticRegression(max_iter=1000)

start = time.time()
log_model.fit(X_train, y_train)
log_time = time.time() - start

y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Report\n")
print(classification_report(y_test, y_pred_log))

# log_metrics = {
#     "Accuracy": accuracy_score(y_test, y_pred_log),
#     "Precision": precision_score(y_test, y_pred_log),
#     "Recall": recall_score(y_test, y_pred_log),
#     "F1": f1_score(y_test, y_pred_log),
#     "Time": log_time
# }

# -------------------------------
# Logistic GridSearch
# -------------------------------
log_params = {
    "C": [0.01, 0.1, 1, 10, 100]
}

log_grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    log_params,
    cv=5,
    scoring="accuracy"
)

log_grid.fit(X_train, y_train)

best_log = log_grid.best_estimator_
print("Best Logistic Params:", log_grid.best_params_)


# -------------------------------
# SVM Kernels
# -------------------------------
kernels = ["linear", "rbf", "poly", "sigmoid"]
svm_results = []

for k in kernels:
    svm = SVC(kernel=k, probability=True)

    start = time.time()
    svm.fit(X_train, y_train)
    t = time.time() - start

    pred = svm.predict(X_test)

    svm_results.append([
        k.upper(),
        accuracy_score(y_test, pred),
        f1_score(y_test, pred),
        t
    ])

    print(f"\nSVM ({k.upper()}) Report\n")
    print(classification_report(y_test, pred))


# -------------------------------
# SVM GridSearch
# -------------------------------
svm_params = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"],
    "kernel": ["linear", "rbf"]
}

svm_grid = GridSearchCV(
    SVC(probability=True),
    svm_params,
    cv=5,
    scoring="accuracy"
)

svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_
print("Best SVM Params:", svm_grid.best_params_)


# -------------------------------
# Cross Validation
# -------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

log_cv = cross_val_score(best_log, X_scaled, y, cv=kf)
svm_cv = cross_val_score(best_svm, X_scaled, y, cv=kf)

print("Logistic CV:", log_cv)
print("SVM CV:", svm_cv)


# -------------------------------
# Confusion + ROC
# -------------------------------
def plot_all(model, X, y, title):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(title)
    plt.show()

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title + " ROC")
    plt.legend()
    plt.show()

# -------------------------------
# Plots
# -------------------------------
plot_all(best_log,X_test,y_test,"Logistic Regression")
plot_all(best_svm,X_test,y_test,"SVM Best Model")

# -------------------------------
# Simple comparison plot
# -------------------------------

labels = ["Logistic", "SVM"]
scores = [log_cv.mean(), svm_cv.mean()]

plt.figure(figsize=(5,4))
plt.bar(labels, scores)
plt.title("Average CV Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
