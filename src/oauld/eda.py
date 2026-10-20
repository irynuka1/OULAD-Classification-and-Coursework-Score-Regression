"""Exploratory data analysis: summary tables and figures."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config as C
from .data import load_raw
from .evaluation import save_fig

EDA_DIR = C.FIG_DIR / "eda"


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes("number").columns if c not in C.TARGETS and c != C.ID_COL]


def attribute_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summary statistics per attribute type (numeric / nominal / ordinal)."""
    def categorical(cols):
        return pd.DataFrame({
            "non_null": [df[c].notna().sum() for c in cols],
            "unique": [df[c].nunique() for c in cols],
            "most_frequent": [df[c].mode().iloc[0] for c in cols],
            "missing": [df[c].isna().sum() for c in cols]}, index=cols)

    return {"numeric": df[numeric_columns(df)].describe().T.round(2),
            "nominal": categorical(C.NOMINAL_COLS), "ordinal": categorical(C.ORDINAL_COLS)}


def run() -> None:
    df, _, _ = load_raw()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    tables = attribute_tables(df)
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(C.METRICS_DIR / f"eda_{name}_summary.csv")

    # 01 boxplots of selected numeric attributes
    cols = ["studied_credits", "total_clicks_early", "mean_score_early", C.TARGET_REG, "date_registration"]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.5))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color="steelblue")
        ax.set_title(col, fontsize=10)
    fig.suptitle("Boxplots of selected numeric attributes", y=1.02)
    save_fig(fig, EDA_DIR / "01_boxplot_numeric.png")

    # 02 distributions of nominal and ordinal attributes
    for tag, cols, orders, color in [("nominal", C.NOMINAL_COLS, {}, "steelblue"),
                                     ("ordinal", C.ORDINAL_COLS, C.ORDINAL_ORDERS, "darkorange")]:
        fig, axes = plt.subplots(1, len(cols), figsize=(4.8 * len(cols), 4.5))
        for ax, col in zip(axes, cols):
            order = orders.get(col, df[col].value_counts().index)
            sns.countplot(data=df, y=col, order=order, ax=ax, color=color)
            ax.set_title(col, fontsize=10)
            ax.set_ylabel("")
        fig.suptitle(f"Distribution of {tag} attributes", y=1.02)
        save_fig(fig, EDA_DIR / f"02_distribution_{tag}.png")

    # 03 missing values
    missing = (df.isna().mean() * 100)
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(missing.index, missing.values, color="coral")
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing values per attribute")
    save_fig(fig, EDA_DIR / "03_missing_values.png")

    # 04 class balance
    counts = df[C.TARGET_CLF].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, f"{v}\n({v / len(df):.1%})", ha="center")
    ax.set_title("Class distribution of final_result")
    ax.set_ylabel("Students")
    ax.set_xlabel("")
    save_fig(fig, EDA_DIR / "04_class_balance.png")

    # 05 correlation heatmap
    corr_cols = ["studied_credits", "mean_score_early", "weighted_mean_score_early", C.TARGET_REG,
                 "total_clicks_early", "active_days_early", "clicks_forumng", "clicks_homepage",
                 "clicks_oucontent", "date_registration", "submission_rate_early", "refs_forumng",
                 "n_assessments_early", "macro_stats"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Pearson correlation of numeric attributes")
    save_fig(fig, EDA_DIR / "05_correlation_heatmap.png")

    # 06 numeric attribute vs. classification target (Kruskal-Wallis)
    order = counts.index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x=C.TARGET_CLF, y="mean_score_early", order=order, hue=C.TARGET_CLF,
                legend=False, palette="Set2", ax=ax)
    ax.set_title("mean_score_early by final_result")
    save_fig(fig, EDA_DIR / "06_numeric_vs_classification.png")
    h, p = stats.kruskal(*[g["mean_score_early"].dropna() for _, g in df.groupby(C.TARGET_CLF)])
    print(f"Kruskal-Wallis mean_score_early ~ final_result: H={h:.1f}, p={p:.2e}")

    # 07 categorical attribute vs. classification target (chi-square)
    table = pd.crosstab(df["highest_education"], df[C.TARGET_CLF]).loc[C.ORDINAL_ORDERS["highest_education"]]
    chi2, p, dof, _ = stats.chi2_contingency(table)
    print(f"Chi-square highest_education ~ final_result: chi2={chi2:.1f}, dof={dof}, p={p:.2e}")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    table.div(table.sum(axis=1), axis=0).plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
    ax.set_ylabel("Share of students")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="final_result", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title("final_result by highest_education")
    save_fig(fig, EDA_DIR / "07_categorical_vs_classification.png")

    # 08 numeric attribute vs. regression target
    sample = df.sample(3000, random_state=C.SEED)
    valid = df["mean_score_early"].notna()
    r, _ = stats.pearsonr(df.loc[valid, "mean_score_early"], df.loc[valid, C.TARGET_REG])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sample["mean_score_early"], sample[C.TARGET_REG], s=8, alpha=0.3)
    ax.set_xlabel("mean_score_early")
    ax.set_ylabel(C.TARGET_REG)
    ax.set_title(f"Early score vs. final score (Pearson r = {r:.2f})")
    save_fig(fig, EDA_DIR / "08_numeric_vs_regression.png")

    # 09 categorical attribute vs. regression target
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="code_module", y=C.TARGET_REG, hue="code_module", legend=False,
                palette="viridis", ax=ax)
    ax.set_title("final_coursework_score by code_module")
    save_fig(fig, EDA_DIR / "09_categorical_vs_regression.png")
    print(f"EDA figures written to {EDA_DIR}")
