"""Paths, column groups and global constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
PRED_DIR = REPORTS_DIR / "predictions"

TRAIN_CSV = RAW_DIR / "CA_OUALD_train.csv"
TEST_CSV = RAW_DIR / "CA_OUALD_test.csv"          # labelled hold-out set
PRIVATE_CSV = RAW_DIR / "CA_private_test.csv"     # unlabelled Kaggle set

SEED = 42
N_FOLDS = 5

TARGET_CLF = "final_result"
TARGET_REG = "final_coursework_score"
TARGETS = [TARGET_CLF, TARGET_REG]
ID_COL = "id"

NOMINAL_COLS = ["code_module", "code_presentation", "gender", "region", "disability"]
ORDINAL_ORDERS = {
    "highest_education": ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                          "HE Qualification", "Post Graduate Qualification"],
    "imd_band": ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%",
                 "60-70%", "70-80%", "80-90%", "90-100%"],
    "age_band": ["0-35", "35-55", "55<="],
    "clicks_freq_init": ["low", "mid", "high"],
}
ORDINAL_COLS = list(ORDINAL_ORDERS)
CATEGORICAL_COLS = NOMINAL_COLS + ORDINAL_COLS

MISSING_TOKEN = "missing"

# Heavy-tailed columns winsorised (IQR rule) for the linear baselines
OUTLIER_COLS = ["studied_credits", "total_clicks_early", "active_days_early", "clicks_dataplus",
                "clicks_forumng", "clicks_homepage", "clicks_oucontent", "clicks_subpage",
                "refs_forumng", "num_of_prev_attempts"]
