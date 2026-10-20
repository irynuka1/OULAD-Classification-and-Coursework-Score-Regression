"""Data loading and stateless feature construction."""
import pandas as pd

from . import config as C


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, labelled test, private test) as provided."""
    return (pd.read_csv(C.TRAIN_CSV), pd.read_csv(C.TEST_CSV), pd.read_csv(C.PRIVATE_CSV))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning shared by every model (no statistics are learned here).

    - drops ids and targets
    - missing ``clicks_*`` / ``refs_forumng`` mean "no activity" -> 0
    - ``was_unregistered`` flags a non-missing ``date_unregistration`` (99.6 % of
      those students withdrew); ``score_missing`` flags students with no early score
    - categorical NaNs become an explicit ``"missing"`` level and every categorical
      column is a plain string column
    Remaining numeric NaNs are left to the model (GBDTs handle them natively; the
    linear baselines impute them inside their pipeline, fitted on training folds only).
    """
    out = df.drop(columns=[c for c in [C.ID_COL, *C.TARGETS] if c in df.columns]).copy()

    out["was_unregistered"] = out["date_unregistration"].notna().astype(int)
    out["score_missing"] = out["mean_score_early"].isna().astype(int)

    activity_cols = [c for c in out.columns
                     if c.startswith("clicks_") and c not in C.ORDINAL_COLS] + ["refs_forumng"]
    out[activity_cols] = out[activity_cols].fillna(0)

    for col in C.CATEGORICAL_COLS:
        out[col] = out[col].fillna(C.MISSING_TOKEN).astype(str)
    return out
