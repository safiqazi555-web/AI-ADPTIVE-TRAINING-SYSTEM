"""Evaluate the ANN against baselines (Sections 5.5 and 5.7 of the report).

Run:  python evaluate.py        (after  python model.py)
Creates in ./reports : model_comparison.csv, metrics.json, confusion_matrix_*.png,
                       training_curves.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.tree import DecisionTreeClassifier

from config import LEVELS, RANDOM_STATE, REPORT_DIR
from data_preprocessing import clean, load_dataset, load_scaler, split_data, transform
from model import HISTORY_PATH, load_model


def _metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": p, "Recall": r, "F1-score": f1}


def _plot_cm(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=range(len(LEVELS))),
                           display_labels=LEVELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(history["loss"], label="Training"); axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch"); axes[0].legend()
    axes[1].plot(history["accuracy"], label="Training"); axes[1].plot(history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate() -> pd.DataFrame:
    REPORT_DIR.mkdir(exist_ok=True)
    df = clean(load_dataset())
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df)
    scaler, bounds = load_scaler()
    X_tr_s, X_te_s = transform(X_tr, scaler, bounds), transform(X_te, scaler, bounds)

    ann = load_model()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_tr_s, y_tr),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=6,
                                                random_state=RANDOM_STATE).fit(X_tr_s, y_tr),
        "ANN (MLP)": ann,
    }
    rows, details = [], {}
    for name, m in models.items():
        pred = m.predict(X_te_s)
        rows.append({"Model": name, **_metrics(y_te, pred)})
        details[name] = classification_report(y_te, pred, target_names=LEVELS, output_dict=True, zero_division=0)
        _plot_cm(y_te, pred, name, REPORT_DIR / f"confusion_matrix_{name.split()[0].lower()}.png")

    table = pd.DataFrame(rows).round(4)
    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(details, indent=2))
    if HISTORY_PATH.exists():
        _plot_curves(json.loads(HISTORY_PATH.read_text()), REPORT_DIR / "training_curves.png")
    print("\nModel comparison on the unseen test set (Table 8 of the report):\n")
    print(table.to_string(index=False))
    print("\nANN classification report:\n")
    print(classification_report(y_te, ann.predict(X_te_s), target_names=LEVELS, zero_division=0))
    print(f"Files saved in {REPORT_DIR}")
    return table


if __name__ == "__main__":
    evaluate()
