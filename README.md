# OULAD – Final Result Classification & Coursework Score Regression

Using each student's behaviour during the early part of a module, we predict:


| Task           | Target                                                 | Type                 | Selection metric    |
| -------------- | ------------------------------------------------------ | -------------------- | ------------------- |
| Classification | `final_result` (Withdrawn / Fail / Pass / Distinction) | 4 imbalanced classes | macro-F1            |
| Regression     | `final_coursework_score` (0–100)                       | continuous           | R² (also MAE, RMSE) |


## Results

Models are fitted on `CA_OUALD_train.csv` and scored on the labelled hold-out `CA_OUALD_test.csv`
(5-fold CV on the training set is reported next to it).

**Classification**


| Model                           | CV F1 (macro) | Test accuracy | Test F1 (macro) |
| ------------------------------- | ------------- | ------------- | --------------- |
| Decision Tree                   | 0.666         | 0.657         | 0.676           |
| Logistic Regression             | 0.683         | 0.673         | 0.690           |
| Random Forest                   | 0.723         | 0.746         | 0.736           |
| CatBoost                        | 0.726         | 0.754         | 0.741           |
| LightGBM                        | 0.709         | 0.763         | 0.721           |
| **Blend + tuned class weights** | **0.727**     | 0.758         | 0.740           |


**Regression**


| Model                           | CV R²     | Test MAE | Test RMSE | Test R²   |
| ------------------------------- | --------- | -------- | --------- | --------- |
| Linear Regression               | 0.826     | 5.07     | 7.16      | 0.808     |
| Decision Tree                   | 0.815     | 5.11     | 7.30      | 0.800     |
| Random Forest                   | 0.829     | 4.89     | 7.07      | 0.813     |
| CatBoost                        | 0.834     | 4.87     | 6.98      | 0.817     |
| LightGBM                        | 0.834     | 4.87     | 6.99      | 0.817     |
| **Blend (CatBoost + LightGBM)** | **0.835** | **4.85** | **6.97**  | **0.818** |


Full tables: `reports/metrics/`. The gains of the final ensembles over a well-tuned single CatBoost are
small (within one CV standard deviation): the data appears to be close to its noise ceiling
(the main residual error is Fail ↔ Pass confusion). See `docs/report.md` for the analysis.

## Project layout

```
.
├── data/oauld_data.zip       # zipped dataset (unzip into data/raw/, which is git-ignored)
├── data/raw/                 # CA_OUALD_train / CA_OUALD_test (labelled) / CA_private_test (Kaggle)
├── src/oauld/                # the library
│   ├── config.py             #   paths, column groups, seeds
│   ├── data.py               #   loading + stateless feature construction
│   ├── preprocessing.py      #   leak-free sklearn pipelines (imputing, IQR clipping, encoding, scaling)
│   ├── models.py             #   model factories (baselines, CatBoost, LightGBM)
│   ├── evaluation.py         #   metrics, cross-validation, class-weight tuning, plots
│   ├── eda.py                #   EDA tables and figures
│   ├── classification.py     #   classification experiment
│   ├── regression.py         #   regression experiment
│   └── cli.py                #   `python -m oauld ...`
├── notebooks/01_eda.ipynb    # exploratory analysis
├── reports/
│   ├── figures/{eda,classification,regression}/
│   ├── metrics/              # comparison tables (csv + md), decision-tree grid, class weights
│   └── predictions/          # predictions_clf.csv, predictions_reg.csv (Kaggle format)
├── docs/                     # report.md, assignment brief, original (Romanian) report
├── archive/                  # original notebooks and predictions, kept for reference
├── Makefile · pyproject.toml · requirements.txt
```



## Usage

The dataset is versioned as `data/oauld_data.zip`; unzip it first (the CSVs are git-ignored):

```bash
unzip -j data/oauld_data.zip -d data/raw
pip install -r requirements.txt        # or: pip install -e .
make eda                               # tables + figures
make dev                               # train on train, evaluate on the labelled hold-out
make submit                            # train on train + hold-out, predict CA_private_test
```

Equivalent: `PYTHONPATH=src python -m oauld {eda|classify|regress|all} [--mode dev|submit] [--gpu]`.
`--gpu` trains CatBoost on GPU 0 (CPU by default, for reproducibility). The seed is fixed to 42.

Modes:

- **dev** – fit on `CA_OUALD_train`, report CV and hold-out metrics, write figures/tables.
- **submit** – fit on train **plus** the labelled hold-out (more data, nothing left to evaluate on) and
write `reports/predictions/predictions_{clf,reg}.csv` with the `id` of `CA_private_test`.



## Method in short

1. **Features** (`data.py`): missing `clicks_`* → 0 (no activity); `was_unregistered` and `score_missing`
  flags; categorical NaN → explicit `"missing"` level. No statistic is learned here.
2. **Pipelines** (`preprocessing.py`): every fitted step (median imputation, IQR winsorising of 10 heavy-tailed
  columns, one-hot / ordinal encoding, standardisation) lives inside an sklearn `Pipeline`, so it is fitted on
   training folds only. Gradient-boosting models use categoricals natively and need no encoding.
3. **Models**: Decision Tree, Random Forest, Logistic/Linear/Ridge/Lasso as baselines; CatBoost and LightGBM as
  main models; final prediction = average of CatBoost and LightGBM.
4. **Classification decision rule**: per-class probability multipliers tuned on out-of-fold probabilities to
  maximise macro-F1 and then frozen.
5. **Evaluation**: 5-fold CV (stratified for classification) plus an untouched labelled hold-out.
