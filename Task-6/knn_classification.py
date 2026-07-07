"""
K-Nearest Neighbors (KNN) Classification
------------------------------------------
Goal: understand and implement KNN for a classification problem.

Dataset : Iris (comes built-in with scikit-learn)
Steps   :
    1. Load and explore the data
    2. Split into train/test sets
    3. Try KNN with different values of K
    4. Evaluate using accuracy and a confusion matrix
    5. Visualize the decision boundary

Libraries: scikit-learn, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# -------------------------------------------------
# 1. Load the data
# -------------------------------------------------
iris = load_iris()

# Using only 2 features (petal length & petal width) so we can plot
# the decision boundary in 2D. Feel free to use all 4 features if you
# don't need the visualization.
X = iris.data[:, [2, 3]]
y = iris.target

df = pd.DataFrame(X, columns=["petal_length", "petal_width"])
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

print("First 5 rows of the dataset:")
print(df.head())
print("\nClass distribution:")
print(df["species"].value_counts())


# -------------------------------------------------
# 2. Train/test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# KNN uses distances, so features need to be on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------
# 3. Try different values of K
# -------------------------------------------------
k_values = list(range(1, 16))
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k:2d}  ->  Accuracy = {acc:.4f}")

# Plot accuracy vs K so we can pick a good value
plt.figure(figsize=(7, 5))
plt.plot(k_values, accuracies, marker="o")
plt.title("Accuracy vs K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Test Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.close()

# Pick the best K found above
best_k = k_values[accuracies.index(max(accuracies))]
print(f"\nBest K found: {best_k} (Accuracy = {max(accuracies):.4f})")


# -------------------------------------------------
# 4. Final model + evaluation
# -------------------------------------------------
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred = final_knn.predict(X_test_scaled)

final_acc = accuracy_score(y_test, y_pred)
print(f"\nFinal Model Accuracy (K={best_k}): {final_acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()


# -------------------------------------------------
# 5. Visualize the decision boundary
# -------------------------------------------------
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # step size in the mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FFD9D9", "#D9FFD9", "#D9D9FF"])
    cmap_bold = ["red", "green", "blue"]

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

    for i, species in enumerate(iris.target_names):
        plt.scatter(
            X[y == i, 0], X[y == i, 1],
            c=cmap_bold[i], label=species,
            edgecolor="k", s=40
        )

    plt.title(title)
    plt.xlabel("Petal Length (scaled)")
    plt.ylabel("Petal Width (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150)
    plt.close()


# Train on the full scaled dataset just for a nicer looking boundary plot
X_scaled_full = scaler.fit_transform(X)
final_knn.fit(X_scaled_full, y)
plot_decision_boundary(final_knn, X_scaled_full, y, f"KNN Decision Boundary (K={best_k})")

print("\nSaved plots: accuracy_vs_k.png, confusion_matrix.png, decision_boundary.png")
