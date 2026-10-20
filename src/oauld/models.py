"""Model factories for the classification and regression tasks."""
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config as C
from .preprocessing import CategoricalCaster, linear_preprocessor, tree_preprocessor

SEED = C.SEED


def _tree_pipe(est, X):
    return Pipeline([("prep", tree_preprocessor(X)), ("model", est)])


def _linear_pipe(est, X):
    return Pipeline([("prep", linear_preprocessor(X)), ("model", est)])


def _device(gpu: bool) -> dict:
    return {"task_type": "GPU", "devices": "0"} if gpu else {"thread_count": -1}


# --------------------------------------------------------------------------- classification
def catboost_clf(gpu=False):
    return CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03, auto_class_weights="SqrtBalanced",
        cat_features=C.CATEGORICAL_COLS, random_seed=SEED, verbose=0, **_device(gpu))


def lightgbm_clf():
    return Pipeline([("cast", CategoricalCaster(C.CATEGORICAL_COLS)),
                     ("model", lgb.LGBMClassifier(
                         n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                         random_state=SEED, n_jobs=-1, verbose=-1))])


def classification_baselines(X) -> dict:
    """Interpretable reference models (+ Random Forest), keyed by display name."""
    return {
        "Decision Tree": _tree_pipe(DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=50, class_weight="balanced", random_state=SEED), X),
        "Random Forest": _tree_pipe(RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=5, class_weight="balanced",
            random_state=SEED, n_jobs=-1), X),
        "Logistic Regression": _linear_pipe(LogisticRegression(
            max_iter=2000, class_weight="balanced"), X),
    }


def decision_tree_clf_for_search(X):
    return _tree_pipe(DecisionTreeClassifier(class_weight="balanced", random_state=SEED), X)


# --------------------------------------------------------------------------- regression
def catboost_reg(gpu=False):
    return CatBoostRegressor(
        iterations=1500, depth=6, learning_rate=0.03, cat_features=C.CATEGORICAL_COLS,
        random_seed=SEED, verbose=0, **_device(gpu))


def lightgbm_reg():
    return Pipeline([("cast", CategoricalCaster(C.CATEGORICAL_COLS)),
                     ("model", lgb.LGBMRegressor(
                         n_estimators=800, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                         random_state=SEED, n_jobs=-1, verbose=-1))])


def regression_baselines(X) -> dict:
    return {
        "Linear Regression": _linear_pipe(LinearRegression(), X),
        "Ridge": _linear_pipe(Ridge(alpha=10.0), X),
        "Lasso": _linear_pipe(Lasso(alpha=0.05, max_iter=10000), X),
        "Decision Tree": _tree_pipe(DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=20, random_state=SEED), X),
        "Random Forest": _tree_pipe(RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=SEED, n_jobs=-1), X),
    }
