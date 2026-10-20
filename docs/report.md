# Report – Introduction to Machine Learning (OULAD)

Adapted and extended from the original Romanian report (`docs/original_report_ro.pdf`).
All numbers come from `make dev` and are stored in `reports/metrics/`; figures are in `reports/figures/`.

## 1. Setup and reproducibility

- Seed 42 everywhere. Run `make dev` for experiments and `make submit` for Kaggle predictions
  (see the README for the two modes).
- Splits: `CA_OUALD_train.csv` is the training set; `CA_OUALD_test.csv` (labelled) is an untouched hold-out;
  `CA_private_test.csv` has no labels. Model comparison uses 5-fold CV on the training set and the hold-out.

## 2. Exploratory data analysis

Reproduced by `notebooks/01_eda.ipynb` / `make eda`.

### 2.1 Attribute types
- **Numeric:** 33 columns (clicks, scores, dates, credits, …). Summary table: `reports/metrics/eda_numeric_summary.csv`.
- **Nominal (5):** `code_module`, `code_presentation`, `gender`, `region`, `disability`.
- **Ordinal (4):** `highest_education`, `imd_band`, `age_band`, `clicks_freq_init`.

Observations from the boxplots and histograms:
- `studied_credits` is concentrated with a few high outliers (very active students); `total_clicks_early`
  has strong upper outliers.
- `mean_score_early` has outliers near 0 (students who skipped the assessments); `final_coursework_score`
  has many extreme low values.
- Most students register before the module starts (`date_registration` < 0).
- Gender is balanced; most students come from the south/south-east of the UK; few declare a disability;
  `A Level or Equivalent` dominates `highest_education`; most students are under 35.

Missing values (`03_missing_values.png`): `date_unregistration` 81 %, `clicks_glossary` / `clicks_externalquiz`
~30 %, `imd_band` 4 %, the rest < 1 %.

### 2.2 Class balance
Pass 47.8 %, Fail 22.1 %, Withdrawn 18.9 %, Distinction 11.2 %. A model always predicting *Pass* reaches ~48 %
accuracy, so accuracy is misleading and **macro-F1** is used for selection.

### 2.3 Correlations and target relationships
- `mean_score_early` ↔ `weighted_mean_score_early` r = 0.97 and `clicks_forumng` ↔ `refs_forumng` r = 0.99 are redundant;
  `total_clicks_early` ↔ `active_days_early` r = 0.78.
- Numeric → classification: Kruskal–Wallis on `mean_score_early` across classes is highly significant
  (Distinction has the highest median, Withdrawn/Fail the lowest).
- Categorical → classification: chi-square between `highest_education` and `final_result` gives
  p ≈ 1e-134 (higher education ⇒ more distinctions).
- Numeric → regression: `mean_score_early` correlates ≈ 0.8 with the final score.
- Categorical → regression: `code_module` shifts the median score (modules differ in difficulty).
- Not in the original report but important: when `date_unregistration` is present, the student *withdrew*
  in 99.6 % of cases.

## 3. Preprocessing

| Step | Original version | Current version |
|---|---|---|
| Missing `clicks_*`, `refs_forumng` | 0 | 0 (no activity) |
| `date_unregistration` (81 % missing) | filled with module length | kept as NaN + `was_unregistered` flag |
| `imd_band` | mode of train | explicit `"missing"` level (trees) / median of ordinal code (linear) |
| Early-score columns | median | native NaN handling (trees) / median (linear) + `score_missing` flag |
| Outliers | IQR×1.5 capping, 10 columns, global | same rule, only inside the linear pipeline, fitted per fold; degenerate (IQR = 0) columns are left untouched |
| Encoding | one-hot nominal, ordinal for ordinals, label-encoded target | same (sklearn `ColumnTransformer`) |
| Scaling | `StandardScaler` | `StandardScaler`, linear models only |
| Leakage control | statistics from the train split | all statistics fitted inside CV folds via pipelines |

Redundant columns were *not* removed: the original experiment showed worse results without them.
Engineered ratios (clicks per active day, click shares, …) were tried with CatBoost and gave ≤ 0.001 change, so they were dropped.

## 4. Classification

Results (hold-out; CV on training set): see `reports/metrics/classification_comparison.md`.

| Model | CV F1 (macro) | Acc. | Precision | Recall | F1 (macro) |
|---|---|---|---|---|---|
| Decision Tree | 0.666 ± 0.008 | 0.657 | 0.671 | 0.735 | 0.676 |
| Logistic Regression | 0.683 ± 0.007 | 0.673 | 0.684 | 0.751 | 0.690 |
| Random Forest | 0.723 ± 0.003 | 0.746 | 0.732 | 0.743 | 0.736 |
| CatBoost | 0.726 ± 0.003 | 0.754 | 0.741 | 0.744 | 0.741 |
| LightGBM | 0.709 ± 0.007 | 0.763 | 0.768 | 0.698 | 0.721 |
| Blend (CatBoost+LightGBM) | 0.722 ± 0.004 | 0.766 | 0.762 | 0.725 | 0.740 |
| Blend + class weights | 0.727 ± 0.002 | 0.758 | 0.748 | 0.738 | 0.740 |

(The CV score of the weighted blend is slightly optimistic because the weights are tuned on the same out-of-fold predictions.)

### 4.1 Baseline: decision tree
A 3-fold grid search over `max_depth` ∈ {3…20}, `min_samples_leaf` ∈ {1…100} and `criterion` (`reports/metrics/decision_tree_grid.csv`)
shows the usual trade-off: shallow trees underfit, deep trees with tiny leaves overfit. Results plateau around depth 8–15;
the best configuration is Gini (0.667 vs 0.665 for entropy – no meaningful difference). The reference model uses
depth 10 and `min_samples_leaf` 50, with balanced class weights.

### 4.2 Advanced models
- **Random Forest** (300 trees, depth 20, leaf 5, balanced weights) averages away the variance of single trees; deeper
  trees are affordable because errors of individual trees are decorrelated.
- **Logistic Regression** is kept to show the limit of linear models: with balanced class weights its recall is high
  but precision and accuracy are the lowest, because it over-predicts rare classes.
- **CatBoost** (1500 iterations, depth 6, lr 0.03, `SqrtBalanced` class weights) handles categoricals natively
  and uses ordered boosting; it is the strongest single model. `SqrtBalanced` is gentler than `Balanced` and gave better macro-F1.
- **LightGBM** adds diversity to the ensemble; its probabilities are averaged with CatBoost's.

### 4.3 Analysis
- Boosting > bagging > single tree > linear, but the gap between CatBoost and Random Forest is small.
- The confusion matrices (`reports/figures/classification/confusion_matrices.png`) show that nearly all remaining error is
  Fail ↔ Pass; this looks like missing information (late-course behaviour is not in the features) rather than model weakness.
- Tuned class weights trade a little accuracy for recall on minority classes; on the hold-out the blend is
  statistically tied with CatBoost alone (0.740 vs 0.741), so the ensemble mostly buys stability (lower CV variance).
- Top features (`reports/figures/classification/catboost_importance.png`) are the unregistration information (`was_unregistered`,
  `date_unregistration`), then the early scores, then `code_module` and `submission_rate_early`; individual click columns matter little.

## 5. Regression

R² is the main metric (scale-free); MAE and RMSE are reported in score points.

| Model | CV R² | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| Linear Regression | 0.826 ± 0.004 | 5.07 | 51.29 | 7.16 | 0.808 |
| Ridge | 0.826 ± 0.004 | 5.07 | 51.29 | 7.16 | 0.808 |
| Lasso | 0.826 ± 0.004 | 5.06 | 51.41 | 7.17 | 0.807 |
| Decision Tree | 0.815 ± 0.006 | 5.11 | 53.32 | 7.30 | 0.800 |
| Random Forest | 0.829 ± 0.003 | 4.89 | 49.93 | 7.07 | 0.813 |
| CatBoost | 0.834 ± 0.003 | 4.87 | 48.77 | 6.98 | 0.817 |
| LightGBM | 0.834 ± 0.004 | 4.87 | 48.87 | 6.99 | 0.817 |
| Blend (CatBoost+LightGBM) | 0.835 ± 0.003 | 4.85 | 48.53 | 6.97 | 0.818 |

- **Linear models** are surprisingly strong (R² 0.81) because a few features correlate linearly with the final score.
  L1/L2 regularisation does not help: there is no serious multicollinearity, consistent with the earlier
  observation that dropping redundant features hurts.
- **Learning curves** (`reports/figures/regression/learning_curves.png`): linear regression converges quickly with a constant
  gap (structural bias: engagement vs. score is non-linear); the decision tree keeps improving with data; Random Forest and
  CatBoost show a train/validation gap that shrinks slowly – more data would help a little.
- **Decision tree** predictions are plateaus (one value per leaf); averaging in a forest or boosting removes this
  (`predicted_vs_actual.png`).
- **CatBoost / LightGBM** are the best models; Predictions are clipped to [0, 100].
- Stacking classifier probabilities into the regressor, Huber loss, depth 4/8 and L2 regularisation changed R² by < 0.002.

## 6. Conclusions

- Gradient boosting with native categorical handling is the best family on both tasks, with a modest margin over a tuned
  Random Forest and a large one over linear models and single trees.
- The ensemble and class-weight tuning give small but consistent gains in CV; the hold-out differences are within noise.
  The data seems close to its ceiling: further gains would more likely come from new information (e.g. later-course activity) than
  from more tuning.
- Methodological improvements over the first version (per-fold preprocessing, no evaluation-set leakage in CatBoost, CV
  selection, final fit on all labelled data) make the reported numbers more trustworthy than the absolute increase suggests.
- Possible future work: Optuna-based tuning, temporal features, calibration of class probabilities.
