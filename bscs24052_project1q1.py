import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, f1_score, accuracy_score

data = np.loadtxt("housing.csv")
features = data[:, :-1]   
prices   = data[:, -1]    

features = (features - features.mean(axis=0)) / features.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

def model(x, w, c):
    return np.dot(x, w) + c
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def gradients(x, y, y_hat):
    n = len(y)
    error = y_hat - y
    dw = (2/n) * np.dot(x.T, error)
    dc = (2/n) * np.sum(error)
    return dw, dc
def stop_check(old_w, new_w, eps=1e-4):
    return np.linalg.norm(new_w - old_w) <= eps
def train_model(x, y, lr=0.01, max_steps=4000):
    w = np.zeros(x.shape[1])
    c = 0
    loss_list = []

    for step in range(max_steps):
        y_hat = model(x, w, c)
        loss = mse(y, y_hat)
       
        if np.isnan(loss) or loss > 1e10:
            print(f"lr={lr} diverged at step {step}")
            return w, c, loss_list
        loss_list.append(loss)
        dw, dc = gradients(x, y, y_hat)
        new_w = w - lr * dw
        new_c = c - lr * dc
        if step > 500 and stop_check(w, new_w):
            print(f"lr={lr} converged at step {step}")
            return new_w, new_c, loss_list
        w, c = new_w, new_c
    print(f"lr={lr} finished")
    return w, c, loss_list
rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
print("comparison:")
print("LR        Train MSE     Test MSE")
print("-" * 35)
plt.figure()

best_w, best_c = None, None
for lr in rates:
    w, c, losses = train_model(X_train, y_train, lr)
    if len(losses) == 0:
        continue
    train_error = mse(y_train, model(X_train, w, c))
    test_error  = mse(y_test,  model(X_test,  w, c))
    print(f"{lr:<8} {train_error:>10.4f} {test_error:>10.4f}")
    plt.plot(losses, label=f"lr={lr}")
    if lr == 0.01:  
        best_w, best_c = w, c
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("Loss vs Iterations")
plt.legend()
plt.show()


print("final Model (lr=0.01)")
w, c, _ = train_model(X_train, y_train, 0.01)
print("bias:", round(c, 4))
print("weights:", np.round(w, 4))
pred = model(X_test, w, c)
mse_val = mean_squared_error(y_test, pred)
print("Test MSE:", round(mse_val, 4))
print("sample results:")
for i in range(5):
    print(f"Actual: {y_test[i]:.1f}  Predicted: {pred[i]:.1f}")
errors = np.abs(y_test - pred)
print("errors (>5):", np.sum(errors > 5))

sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)
print("My MSE GOAT:", round(mse_val, 4))
print("Sklearn MSE:", round(mean_squared_error(y_test, sk_pred), 4))
def convert_to_classes(values):
    p1 = np.percentile(values, 33)
    p2 = np.percentile(values, 66)
    classes = np.zeros(len(values))
    classes[values >= p1] = 1
    classes[values >= p2] = 2
    return classes

true_cls = convert_to_classes(y_test)
pred_cls = convert_to_classes(pred)
print("accuracy:", round(accuracy_score(true_cls, pred_cls), 4))
print("F1 score:", round(f1_score(true_cls, pred_cls, average='weighted'), 4))
print("confusion matrix:\n", confusion_matrix(true_cls, pred_cls))