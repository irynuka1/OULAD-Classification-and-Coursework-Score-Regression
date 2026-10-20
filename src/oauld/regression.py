"""Regression of ``final_coursework_score``."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from . import config as C
from . import evaluation as E
from . import models as M
from .data import build_features, load_raw

LC_SIZES = [500, 1000, 2000, 4000, 8000]


def run(mode: str, gpu: bool = False) -> None:
    """``dev``: fit on train, evaluate on the labelled hold-out (CA_OUALD_test).
    ``submit``: fit on train + hold-out, write predictions for the private test set."""
    train, test, private = load_raw()
    dev = mode == "dev"
    fit_df = train if dev else pd.concat([train, test], ignore_index=True)
    X_fit, y_fit = build_features(fit_df), fit_df[C.TARGET_REG].to_numpy()
    X_eval = build_features(test if dev else private)
    y_eval = test[C.TARGET_REG].to_numpy() if dev else None

    members = {"CatBoost": M.catboost_reg(gpu), "LightGBM": M.lightgbm_reg()}
    baselines = M.regression_baselines(X_fit) if dev else {}

    rows, eval_pred, oof_pred = [], {}, {}
    for name, model in {**baselines, **members}.items():
        print(f"[reg] {name}", flush=True)
        oof, folds = E.cross_val_oof(model, X_fit, y_fit, "reg")
        pred = np.clip(E.fresh(model).fit(X_fit, y_fit).predict(X_eval), 0, 100)
        oof_pred[name], eval_pred[name] = oof, pred
        rows.append(_row(name, folds, y_eval, pred))

    blend_oof = np.mean([oof_pred[m] for m in members], axis=0)
    blend_pred = np.mean([eval_pred[m] for m in members], axis=0)
    eval_pred["Blend (CatBoost+LightGBM)"] = blend_pred
    rows.append(_row("Blend (CatBoost+LightGBM)", _fold_r2(blend_oof, y_fit), y_eval, blend_pred))

    if dev:
        results = pd.DataFrame(rows).set_index("Model")
        _save_dev_outputs(results, X_fit, y_fit, X_eval, y_eval, eval_pred, baselines, members, gpu)
    else:
        C.PRED_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": private[C.ID_COL], "prediction": blend_pred}).to_csv(
            C.PRED_DIR / "predictions_reg.csv", index=False)
        print(f"[reg] wrote {C.PRED_DIR / 'predictions_reg.csv'}")


def _fold_r2(oof, y):
    return [r2_score(y[v], oof[v]) for _, v in KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(oof)]


def _row(name, folds, y_eval, pred) -> dict:
    row = {"Model": name, "CV R2": f"{np.mean(folds):.4f} ± {np.std(folds):.4f}"}
    if y_eval is not None:
        row.update({f"Test {k}": v for k, v in E.reg_metrics(y_eval, pred).items()})
    return row


def _save_dev_outputs(results, X_fit, y_fit, X_eval, y_eval, eval_pred, baselines, members, gpu):
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(C.METRICS_DIR / "regression_comparison.csv")
    (C.METRICS_DIR / "regression_comparison.md").write_text(E.to_markdown(results))
    print(results.to_string())

    shown = {k: eval_pred[k] for k in
             ["Linear Regression", "Decision Tree", "Random Forest", "CatBoost", "Blend (CatBoost+LightGBM)"]}
    E.plot_pred_vs_actual(y_eval, shown, C.FIG_DIR / "regression" / "predicted_vs_actual.png",
                          "Predicted vs. actual score on the hold-out set")

    sizes = [s for s in LC_SIZES if s < len(X_fit)] + [len(X_fit)]
    curve_models = {"Linear Regression": baselines["Linear Regression"],
                    "Decision Tree": baselines["Decision Tree"],
                    "Random Forest": baselines["Random Forest"], "CatBoost": members["CatBoost"]}
    curves = {n: E.learning_curve(m, X_fit, y_fit, X_eval, y_eval, sizes) for n, m in curve_models.items()}
    E.plot_learning_curves(curves, sizes, C.FIG_DIR / "regression" / "learning_curves.png")

    cb = M.catboost_reg(gpu).fit(X_fit, y_fit)
    E.plot_importance(X_fit.columns, cb.get_feature_importance(),
                      C.FIG_DIR / "regression" / "catboost_importance.png", "CatBoost feature importance")
