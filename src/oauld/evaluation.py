"""Metrics, cross-validation helpers and result plots."""
import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import KFold, StratifiedKFold

from . import config as C

sns.set_theme(style="whitegrid", palette="muted")


# --------------------------------------------------------------------------- metrics
def clf_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def reg_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


# --------------------------------------------------------------------------- cross-validation
def fresh(model):
    """Unfitted copy of an estimator (CatBoost is not sklearn-clonable, so deep-copy it)."""
    try:
        return clone(model)
    except RuntimeError:
        return copy.deepcopy(model)


def cross_val_oof(model, X, y, task: str, n_folds: int = C.N_FOLDS):
    """Out-of-fold predictions: class probabilities (``clf``) or values (``reg``).

    ``model`` is an unfitted estimator that is cloned for every fold, so no state
    (imputation statistics, scalers, category lists) leaks between folds.
    Returns (oof, per-fold score list) with macro-F1 / R2 as the fold score.
    """
    splitter = (StratifiedKFold if task == "clf" else KFold)(n_folds, shuffle=True,
                                                             random_state=C.SEED)
    oof, scores = None, []
    for fit_idx, val_idx in splitter.split(X, y):
        m = fresh(model).fit(X.iloc[fit_idx], y[fit_idx])
        if task == "clf":
            part = m.predict_proba(X.iloc[val_idx])
            scores.append(f1_score(y[val_idx], part.argmax(1), average="macro"))
        else:
            part = m.predict(X.iloc[val_idx])
            scores.append(r2_score(y[val_idx], part))
        if oof is None:
            oof = np.zeros((len(X),) + part.shape[1:])
        oof[val_idx] = part
    return oof, scores


def tune_class_weights(proba, y, grid=np.linspace(0.5, 2.5, 41), rounds: int = 3):
    """Coordinate search of per-class probability multipliers maximising macro-F1.

    Fitted on out-of-fold probabilities only, then applied unchanged to unseen data.
    """
    w = np.ones(proba.shape[1])
    for _ in range(rounds):
        for k in range(len(w)):
            scores = []
            for s in grid:
                trial = w.copy()
                trial[k] = s
                scores.append(f1_score(y, (proba * trial).argmax(1), average="macro"))
            w[k] = grid[int(np.argmax(scores))]
    return w / w.max()


def learning_curve(model, X_tr, y_tr, X_val, y_val, sizes):
    """Train/validation MAE of ``model`` fitted on growing random subsets of the data."""
    order = np.random.default_rng(C.SEED).permutation(len(X_tr))
    tr_err, val_err = [], []
    for n in sizes:
        idx = order[:n]
        m = fresh(model).fit(X_tr.iloc[idx], y_tr[idx])
        tr_err.append(mean_absolute_error(y_tr[idx], m.predict(X_tr.iloc[idx])))
        val_err.append(mean_absolute_error(y_val, m.predict(X_val)))
    return tr_err, val_err


# --------------------------------------------------------------------------- plots
def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(y_true, preds: dict, class_names, path, title):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.8 * len(preds), 4.4), squeeze=False)
    for ax, (name, pred) in zip(axes[0], preds.items()):
        cm = confusion_matrix(y_true, pred, labels=range(len(class_names)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{name}\nF1 macro = {f1_score(y_true, pred, average='macro'):.3f}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(title, y=1.04)
    save_fig(fig, path)


def plot_pred_vs_actual(y_true, preds: dict, path, title):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.4), squeeze=False)
    for ax, (name, pred) in zip(axes[0], preds.items()):
        ax.scatter(y_true, pred, s=4, alpha=0.2)
        ax.plot([0, 100], [0, 100], "r--", lw=1.2)
        ax.set_title(f"{name}\nR2 = {r2_score(y_true, pred):.3f}, MAE = {mean_absolute_error(y_true, pred):.2f}",
                     fontsize=10)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.suptitle(title, y=1.04)
    save_fig(fig, path)


def plot_learning_curves(curves: dict, sizes, path):
    fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 3.8), squeeze=False)
    for ax, (name, (tr, val)) in zip(axes[0], curves.items()):
        ax.plot(sizes, tr, "o-", label="Train MAE")
        ax.plot(sizes, val, "o-", label="Validation MAE")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=8)
    save_fig(fig, path)


def plot_importance(names, values, path, title, top=20):
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * top + 1))
    ax.barh(np.array(names)[order], np.array(values)[order])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    save_fig(fig, path)


def to_markdown(df: pd.DataFrame, floatfmt=".4f") -> str:
    return df.to_markdown(floatfmt=floatfmt)
