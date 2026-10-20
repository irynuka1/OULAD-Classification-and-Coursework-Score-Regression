"""Model-specific preprocessing pipelines (always fitted on training data only)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns to [Q1 - k*IQR, Q3 + k*IQR] using training quantiles."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        spread = np.where(iqr > 0, self.factor * iqr, np.inf)  # degenerate columns stay untouched
        self.lower_, self.upper_ = q1 - spread, q3 + spread
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, names=None):
        return names


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Cast string categoricals to pandas ``category`` dtype (LightGBM native handling)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].unique()) for c in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in C.CATEGORICAL_COLS]


def _encoders() -> list:
    ordinal = OrdinalEncoder(categories=[C.ORDINAL_ORDERS[c] for c in C.ORDINAL_COLS],
                             handle_unknown="use_encoded_value", unknown_value=np.nan)
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return [("ordinal", ordinal, C.ORDINAL_COLS), ("nominal", onehot, C.NOMINAL_COLS)]


def tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Ordinal + one-hot encoding for scikit-learn trees/forests (NaNs handled natively)."""
    return ColumnTransformer([*_encoders(), ("num", "passthrough", _numeric_cols(X))])


def linear_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputation, IQR winsorising, encoding and standardisation for linear models."""
    ordinal_name, ordinal, ordinal_cols = _encoders()[0]
    ordinal_pipe = Pipeline([("enc", ordinal), ("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler())])
    clipped_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("clip", IQRClipper()), ("sc", StandardScaler())])
    plain_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    plain_cols = [c for c in _numeric_cols(X) if c not in C.OUTLIER_COLS]
    return ColumnTransformer([(ordinal_name, ordinal_pipe, ordinal_cols),
                              _encoders()[1],
                              ("clipped", clipped_pipe, C.OUTLIER_COLS),
                              ("plain", plain_pipe, plain_cols)])
