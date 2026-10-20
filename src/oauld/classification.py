"""Multi-class classification of ``final_result``."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from . import config as C
from . import evaluation as E
from . import models as M
from .data import build_features, load_raw


def decision_tree_study(X, y) -> pd.DataFrame:
    """Grid search over the three decision-tree hyper-parameters studied in the report."""
    grid = {"model__max_depth": [3, 5, 8, 10, 15, 20],
            "model__min_samples_leaf": [1, 5, 20, 50, 100],
            "model__criterion": ["gini", "entropy"]}
    search = GridSearchCV(M.decision_tree_clf_for_search(X), grid, scoring="f1_macro", n_jobs=-1,
                          cv=StratifiedKFold(3, shuffle=True, random_state=C.SEED)).fit(X, y)
    table = pd.DataFrame(search.cv_results_)
    table = table.rename(columns=lambda c: c.replace("param_model__", "")).rename(
        columns={"mean_test_score": "F1 (macro) CV"})
    table = table[["max_depth", "min_samples_leaf", "criterion", "F1 (macro) CV"]]
    return table.sort_values("F1 (macro) CV", ascending=False).reset_index(drop=True)


def run(mode: str, gpu: bool = False) -> None:
    """``dev``: fit on train, evaluate on the labelled hold-out (CA_OUALD_test).
    ``submit``: fit on train + hold-out, write predictions for the private test set."""
    train, test, private = load_raw()
    dev = mode == "dev"
    fit_df = train if dev else pd.concat([train, test], ignore_index=True)
    eval_df = test if dev else None

    X_fit = build_features(fit_df)
    le = LabelEncoder().fit(fit_df[C.TARGET_CLF])
    y_fit = le.transform(fit_df[C.TARGET_CLF])
    classes = list(le.classes_)
    X_eval = build_features(eval_df) if dev else build_features(private)
    y_eval = le.transform(eval_df[C.TARGET_CLF]) if dev else None

    members = {"CatBoost": M.catboost_clf(gpu), "LightGBM": M.lightgbm_clf()}
    baselines = M.classification_baselines(X_fit) if dev else {}

    rows, eval_proba, oof_proba = [], {}, {}
    for name, model in {**baselines, **members}.items():
        print(f"[clf] {name}", flush=True)
        oof, folds = E.cross_val_oof(model, X_fit, y_fit, "clf")
        proba = E.fresh(model).fit(X_fit, y_fit).predict_proba(X_eval)
        oof_proba[name], eval_proba[name] = oof, proba
        rows.append(_row(name, folds, y_eval, proba))

    blend_oof = np.mean([oof_proba[m] for m in members], axis=0)
    blend_eval = np.mean([eval_proba[m] for m in members], axis=0)
    weights = E.tune_class_weights(blend_oof, y_fit)
    print("[clf] tuned class weights:", {c: round(float(w), 2) for c, w in zip(classes, weights)})

    for name, oof, proba in [("Blend (CatBoost+LightGBM)", blend_oof, blend_eval),
                             ("Blend + class weights", blend_oof * weights, blend_eval * weights)]:
        eval_proba[name] = proba
        rows.append(_row(name, _fold_scores(oof, y_fit), y_eval, proba))

    final_proba = eval_proba["Blend + class weights"]
    final_labels = le.inverse_transform(final_proba.argmax(1))

    if dev:
        results = pd.DataFrame(rows).set_index("Model")
        _save_dev_outputs(results, X_fit, y_fit, y_eval, eval_proba, classes, weights, members, gpu)
    else:
        C.PRED_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": private[C.ID_COL], "prediction": final_labels}).to_csv(
            C.PRED_DIR / "predictions_clf.csv", index=False)
        print(f"[clf] wrote {C.PRED_DIR / 'predictions_clf.csv'}")


def _fold_scores(oof_proba, y):
    """Macro-F1 of the pooled out-of-fold predictions, split in the same folds."""
    splitter = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)
    return [f1_score(y[v], oof_proba[v].argmax(1), average="macro") for _, v in splitter.split(oof_proba, y)]


def _row(name, folds, y_eval, proba) -> dict:
    row = {"Model": name, "CV F1 (macro)": f"{np.mean(folds):.4f} ± {np.std(folds):.4f}"}
    if y_eval is not None:
        row.update({f"Test {k}": v for k, v in E.clf_metrics(y_eval, proba.argmax(1)).items()})
    return row


def _save_dev_outputs(results, X_fit, y_fit, y_eval, eval_proba, classes, weights, members, gpu):
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(C.METRICS_DIR / "classification_comparison.csv")
    (C.METRICS_DIR / "classification_comparison.md").write_text(E.to_markdown(results))
    (C.METRICS_DIR / "classification_class_weights.json").write_text(
        json.dumps(dict(zip(classes, map(float, weights))), indent=2))
    print(results.to_string())

    shown = {k: eval_proba[k].argmax(1) for k in
             ["Decision Tree", "Random Forest", "Logistic Regression", "CatBoost", "Blend + class weights"]}
    E.plot_confusion_matrices(y_eval, shown, classes, C.FIG_DIR / "classification" / "confusion_matrices.png",
                              "Confusion matrices on the hold-out set")

    study = decision_tree_study(X_fit, y_fit)
    study.to_csv(C.METRICS_DIR / "decision_tree_grid.csv", index=False)

    cb = M.catboost_clf(gpu).fit(X_fit, y_fit)
    E.plot_importance(X_fit.columns, cb.get_feature_importance(),
                      C.FIG_DIR / "classification" / "catboost_importance.png", "CatBoost feature importance")
