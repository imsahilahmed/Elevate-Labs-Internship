"""
tree_models.py
================
Tree-Based Models for Heart Disease Classification.

Dataset : UCI Heart Disease dataset (data/heart.csv) — 303 patients, 13 clinical
          features, binary target (1 = presence of heart disease, 0 = absence).

Pipeline:
    1. Load & split data
    2. Train a Decision Tree Classifier and visualize it (Graphviz)
    3. Analyze overfitting by sweeping max_depth
    4. Train a Random Forest and compare accuracy against the Decision Tree
    5. Interpret feature importances
    6. Evaluate all models with 5-fold cross-validation

Run:
    python src/tree_models.py

Outputs (saved to outputs/):
    tree_visualization.png
    overfitting_vs_depth.png
    model_comparison.png
    feature_importances.png
"""

import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.25
MAX_DEPTH_RANGE = range(1, 16)
N_TREES_IN_FOREST = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data(path: str):
    """Load the heart disease dataset and split it into train/test sets."""
    df = pd.read_csv(path)

    feature_names = df.columns.drop("target").tolist()
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# 2. Decision tree training + visualization
# ---------------------------------------------------------------------------
def train_decision_tree(X_train, y_train, max_depth=None):
    """Train a Decision Tree Classifier."""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf


def visualize_tree(clf, feature_names, out_path, max_depth_shown=3):
    """Render a Decision Tree to a PNG using Graphviz."""
    dot_data = export_graphviz(
        clf,
        out_file=None,
        max_depth=max_depth_shown,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(filename=out_path, format="png", cleanup=True)
    print(f"[saved] {out_path}.png")


# ---------------------------------------------------------------------------
# 3. Overfitting analysis across tree depth
# ---------------------------------------------------------------------------
def analyze_overfitting(X_train, y_train, X_test, y_test, depths):
    """Track train/test/CV accuracy as max_depth increases; return best depth."""
    train_scores, test_scores, cv_scores = [], [], []

    for depth in depths:
        clf = train_decision_tree(X_train, y_train, max_depth=depth)
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
        cv_scores.append(cross_val_score(clf, X_train, y_train, cv=5).mean())

    best_depth = list(depths)[int(np.argmax(cv_scores))]

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, marker="o", label="Train accuracy")
    plt.plot(depths, test_scores, marker="o", label="Test accuracy")
    plt.plot(depths, cv_scores, marker="o", linestyle="--", label="5-fold CV accuracy")
    plt.axvline(best_depth, color="gray", linestyle=":", label=f"Best depth = {best_depth}")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree: Accuracy vs. Tree Depth (Overfitting Check)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "overfitting_vs_depth.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

    return best_depth, train_scores, test_scores, cv_scores


# ---------------------------------------------------------------------------
# 4. Random Forest + model comparison
# ---------------------------------------------------------------------------
def train_random_forest(X_train, y_train, n_estimators=N_TREES_IN_FOREST):
    """Train a Random Forest Classifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def compare_models(models: dict, X_train, y_train, X_test, y_test):
    """Print & plot train/test accuracy for a dict of {name: fitted_model}."""
    results = {}
    for name, model in models.items():
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = (train_acc, test_acc)
        print(f"  {name:35s} train={train_acc:.4f}  test={test_acc:.4f}")

    names = list(results.keys())
    train_vals = [v[0] for v in results.values()]
    test_vals = [v[1] for v in results.values()]

    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, train_vals, width, label="Train accuracy")
    plt.bar(x + width / 2, test_vals, width, label="Test accuracy")
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Model Comparison: Train vs Test Accuracy")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

    return results


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, out_path, top_n=13, title="Feature Importances"):
    """Plot the top-N feature importances of a fitted tree-based model."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    names_sorted = [feature_names[i] for i in order][::-1]
    vals_sorted = [importances[i] for i in order][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(names_sorted, vals_sorted, color="seagreen")
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

    return list(zip([feature_names[i] for i in order], [importances[i] for i in order]))


# ---------------------------------------------------------------------------
# 6. Cross-validation
# ---------------------------------------------------------------------------
def evaluate_with_cv(models: dict, X_train, y_train, cv=5):
    """Run cross_val_score for each model and print mean/std accuracy."""
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv)
        print(f"  {name:35s} mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores, 3)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 0: Load data")
    print("=" * 70)
    X_train, X_test, y_train, y_test, feature_names = load_data(DATA_PATH)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Features: {feature_names}")

    print("\n" + "=" * 70)
    print("STEP 1: Train & visualize a Decision Tree")
    print("=" * 70)
    tree_full = train_decision_tree(X_train, y_train)
    print(f"Unrestricted tree depth: {tree_full.get_depth()}, leaves: {tree_full.get_n_leaves()}")
    visualize_tree(
        tree_full, feature_names, os.path.join(OUTPUT_DIR, "tree_visualization")
    )

    print("\n" + "=" * 70)
    print("STEP 2: Analyze overfitting across max_depth")
    print("=" * 70)
    best_depth, train_scores, test_scores, cv_scores = analyze_overfitting(
        X_train, y_train, X_test, y_test, MAX_DEPTH_RANGE
    )
    print(f"Best max_depth (by 5-fold CV): {best_depth}")
    tree_pruned = train_decision_tree(X_train, y_train, max_depth=best_depth)

    print("\n" + "=" * 70)
    print("STEP 3: Train Random Forest & compare accuracy")
    print("=" * 70)
    forest = train_random_forest(X_train, y_train)
    models = {
        "Decision Tree (unrestricted)": tree_full,
        f"Decision Tree (max_depth={best_depth})": tree_pruned,
        f"Random Forest ({N_TREES_IN_FOREST} trees)": forest,
    }
    compare_models(models, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 70)
    print("STEP 4: Feature importance (Random Forest)")
    print("=" * 70)
    top_features = plot_feature_importance(
        forest,
        feature_names,
        os.path.join(OUTPUT_DIR, "feature_importances.png"),
        title="Random Forest — Feature Importances",
    )
    for name, importance in top_features:
        print(f"  {name:15s} {importance:.4f}")

    print("\n" + "=" * 70)
    print("STEP 5: 5-fold cross-validation")
    print("=" * 70)
    evaluate_with_cv(models, X_train, y_train, cv=5)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
