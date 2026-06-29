import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("wisc_bc_data.csv")

data = data.drop(columns=["id"])
data["diagnosis"] = (data["diagnosis"] == "M").astype(int)
print("total no:", len(data))
print("malignant:", data["diagnosis"].sum())
print("benign:", (data["diagnosis"] == 0).sum())
X = data.drop(columns=["diagnosis"]).values
y = data["diagnosis"].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
def gradients(X, y, p):
    n = len(y)
    error = p - y
    dw = (1/n) * X.T.dot(error)
    db = (1/n) * np.sum(error)
    return dw, db
def is_converged(w_old, w_new):
    return np.linalg.norm(w_new - w_old) <= 1e-4
def train(X, y, lr=0.1, steps=1000):
    w = np.zeros(X.shape[1])
    b = 0
    losses = []
    for i in range(steps):
        probs = sigmoid(X.dot(w) + b)
        losses.append(loss(y, probs))
        dw, db = gradients(X, y, probs)
        new_w = w - lr * dw
        new_b = b - lr * db
        if i > 100 and is_converged(w, new_w):
            print("Converged at:", i+1)
            return new_w, new_b, losses
        w, b = new_w, new_b
    return w, b, losses
print("training")
w, b, losses = train(X_train, y_train)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

probs = sigmoid(X_test.dot(w) + b)
preds = (probs >= 0.5).astype(int)
print("results:")
print("Accuracy :", round(accuracy_score(y_test, preds), 4))
print("Precision:", round(precision_score(y_test, preds), 4))
print("Recall   :", round(recall_score(y_test, preds), 4))
print("F1 Score :", round(f1_score(y_test, preds), 4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("Cross Validation:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accs = []
f1s = []
for i, (train_i, val_i) in enumerate(kf.split(X)):
    w_k, b_k, _ = train(X[train_i], y[train_i])
    p = (sigmoid(X[val_i].dot(w_k) + b_k) >= 0.5).astype(int)
    acc = accuracy_score(y[val_i], p)
    f1  = f1_score(y[val_i], p)
    accs.append(acc)
    f1s.append(f1)
    print(f"Fold {i+1}: Acc={acc:.4f}, F1={f1:.4f}")
print("avg Accuracy:", np.mean(accs))
print("Avg F1:", np.mean(f1s))
sk = LogisticRegression(max_iter=1000)
sk.fit(X_train, y_train)
sk_preds = sk.predict(X_test)
print("Sklearn Comparison:")
print("Our Acc:", round(accuracy_score(y_test, preds), 4))
print("Sk Acc :", round(accuracy_score(y_test, sk_preds), 4))
model = LogisticRegression(C=0.001, max_iter=1000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.3, 0.7, 41)
accs, precs, recs, f1s = [], [], [], []
for t in thresholds:
    p = (probs >= t).astype(int)
    accs.append(accuracy_score(y_test, p))
    precs.append(precision_score(y_test, p, zero_division=0))
    recs.append(recall_score(y_test, p, zero_division=0))
    f1s.append(f1_score(y_test, p, zero_division=0))

plt.plot(thresholds, accs, label="Accuracy")
plt.plot(thresholds, precs, label="Precision")
plt.plot(thresholds, recs, label="Recall")
plt.plot(thresholds, f1s, label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning")
plt.legend() 
plt.grid()
plt.show()
print("chalo hogya")