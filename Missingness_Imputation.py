#!/usr/bin/env python
# coding: utf-8

# In[16]:


#!/usr/bin/env python3
"""
hmda_missingness_monte.py
==================================================
Controlled experiment measuring how missingness
mechanisms (MCAR, MAR, MNAR) interact with
imputation strategies (mean, linear, tree)
to reshape model decision surfaces.

Replicates:
  "What Models Learn from Missing Data:
   Imputation as Feature Engineering"
Rebecca Whitworth | November 2025
"""

# =========================================================
# Imports
# =========================================================
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =========================================================
# Configuration
# =========================================================
SEED = 4242
np.random.seed(SEED)
CHUNK_SIZE = 1_000_000
N_DRAWS = 25

# directory setup
REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "hmda_2022_lar.txt"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(exist_ok=True)

# =========================================================
# Region mapping (Census divisions)
# =========================================================
REGION_MAP = {
    **dict.fromkeys(
        ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "Northeast"
    ),
    **dict.fromkeys(
        ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
        "Midwest",
    ),
    **dict.fromkeys(
        [
            "DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
            "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX",
        ],
        "South",
    ),
    **dict.fromkeys(
        [
            "AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY",
            "AK", "CA", "HI", "OR", "WA",
        ],
        "West",
    ),
}

# =========================================================
# Data loading and cleaning
# =========================================================
def load_hmda(path: Path, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """
    Load HMDA 2022 LAR data filtered to conventional,
    first-lien, site-built originations.
    """
    cols = [
        "lei", "action_taken", "loan_type", "open_end_line_of_credit",
        "lien_status", "loan_amount", "income", "debt_to_income_ratio",
        "applicant_age", "state_code", "construction_method",
        "occupancy_type", "derived_race", "derived_ethnicity", "derived_sex",
    ]
    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(path, sep="|", usecols=cols, chunksize=chunk_size, low_memory=False)
    ):
        print(f"[chunk {i+1:02d}] loading {len(chunk):,} rows", end=" ")

        # numeric coercion for categorical codes
        for c in [
            "loan_type", "open_end_line_of_credit",
            "lien_status", "action_taken", "construction_method",
        ]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        # apply HMDA filters
        chunk = chunk[
            (chunk["loan_type"] == 1)
            & (chunk["lien_status"] == 1)
            & (chunk["open_end_line_of_credit"].isin([2, 1111]))
            & (chunk["action_taken"].isin([1, 3]))
            & (chunk["construction_method"] == 1)
        ].copy()

        # binary outcome
        chunk["denied"] = (chunk["action_taken"] == 3).astype(int)

        # parse categorical ranges
        chunk["dti_numeric"] = chunk["debt_to_income_ratio"].apply(parse_dti)
        chunk["age_numeric"] = chunk["applicant_age"].apply(parse_age)
        chunk["region"] = chunk["state_code"].map(REGION_MAP)
        print(f" → kept {len(chunk):,}")
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"[✓] Total site-built loans loaded: {len(df):,}")
    return df


def parse_dti(x):
    """Convert HMDA DTI bins to midpoints."""
    if pd.isna(x) or x in ["Exempt", "NA"]:
        return np.nan
    lookup = {
        "<20%": 10.0, "20%-<30%": 25.0, "30%-<36%": 33.0,
        "36%-<40%": 38.0, "40%-<50%": 45.0,
        "50%-60%": 55.0, ">60%": 70.0,
    }
    return lookup.get(str(x).strip(), pd.to_numeric(str(x).replace("%", ""), errors="coerce"))


def parse_age(x):
    """Convert HMDA age ranges to midpoints."""
    if pd.isna(x):
        return np.nan
    lookup = {
        "<25": 22.5, "25-34": 29.5, "35-44": 39.5,
        "45-54": 49.5, "55-64": 59.5, "65-74": 69.5, ">74": 77.5,
    }
    return lookup.get(str(x).strip(), pd.to_numeric(str(x), errors="coerce"))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply core cleaning filters and lender sanity checks.
    """
    lender_ok = df.groupby("lei")["denied"].mean().between(0.01, 0.99)
    df = df[df["lei"].isin(lender_ok[lender_ok].index)].copy()
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df.loc[df["income"] > 2000, "income"] = np.nan
    df.loc[df["age_numeric"].between(18, 80, inclusive="neither") == False, "age_numeric"] = np.nan
    print(f"[✓] After cleaning: {len(df):,} loans / {df['lei'].nunique()} lenders")
    return df

# =========================================================
# Model evaluation utilities
# =========================================================
def curvature(probs: np.ndarray) -> float:
    """Mean absolute 2nd derivative of predicted probabilities."""
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    dp = np.gradient(p)
    ddp = np.gradient(dp)
    return float(np.mean(np.abs(ddp)))


def eval_model(model, Xtr, ytr, Xte, yte):
    """Fit model and return (AUC, R², curvature)."""
    model.fit(Xtr, ytr)
    preds = (
        model.predict_proba(Xte)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(Xte)
    )
    return roc_auc_score(yte, preds), r2_score(yte, preds), curvature(preds)


# =========================================================
# Missingness mechanisms
# =========================================================
def apply_missingness(X, rate, mechanism="MCAR", target_col="dti_numeric", rng=None):
    """
    Apply one of three missingness mechanisms to target column.

    Parameters
    ----------
    X : DataFrame
        Feature matrix
    rate : float
        Proportion to set missing
    mechanism : str
        'MCAR', 'MAR', 'MNAR', or 'INCOME_MCAR'
    target_col : str
        Column to corrupt
    rng : np.random.RandomState
        Random generator for reproducibility
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    n = len(X)

    if mechanism in ["MCAR", "INCOME_MCAR"]:
        mask = rng.rand(n) < rate

    elif mechanism == "MAR":
        z = (X["income"] - X["income"].mean()) / X["income"].std()
        p_missing = rate * (1 / (1 + np.exp(-z)))
        mask = rng.rand(n) < p_missing.to_numpy()

    elif mechanism == "MNAR":
        z = (X[target_col] - X[target_col].mean()) / X[target_col].std()
        p_missing = rate * (1 / (1 + np.exp(-z)))
        mask = rng.rand(n) < p_missing.to_numpy()

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    return mask


# =========================================================
# Imputation strategies
# =========================================================
def impute_mean(Xtr, Xte, col):
    """Mean imputation."""
    mean_val = Xtr[col].mean()
    Xtr[col].fillna(mean_val, inplace=True)
    Xte[col].fillna(mean_val, inplace=True)


def impute_linear(Xtr, Xte, col):
    """Linear regression imputation using observed covariates."""
    notna = Xtr[col].notna()
    if notna.sum() < 10:
        impute_mean(Xtr, Xte, col)
        return
    reg = LinearRegression()
    reg.fit(Xtr.loc[notna].drop(columns=[col]), Xtr.loc[notna, col])
    Xtr.loc[Xtr[col].isna(), col] = reg.predict(Xtr.loc[Xtr[col].isna()].drop(columns=[col]))
    Xte.loc[Xte[col].isna(), col] = reg.predict(Xte.loc[Xte[col].isna()].drop(columns=[col]))


def impute_tree(Xtr, Xte, col, seed=SEED):
    """Gradient boosting imputation."""
    notna = Xtr[col].notna()
    if notna.sum() < 10:
        impute_mean(Xtr, Xte, col)
        return
    reg = GradientBoostingRegressor(random_state=seed)
    reg.fit(Xtr.loc[notna].drop(columns=[col]), Xtr.loc[notna, col])
    Xtr[col].fillna(pd.Series(reg.predict(Xtr.drop(columns=[col])), index=Xtr.index), inplace=True)
    Xte[col].fillna(pd.Series(reg.predict(Xte.drop(columns=[col])), index=Xte.index), inplace=True)


# =========================================================
# Monte Carlo missingness sweep
# =========================================================
def run_missingness_sweep(
    X_full,
    y,
    mechanism="MCAR",
    n_iter=1,
    rates=None,
    seed=SEED,
):
    """
    Monte Carlo sweep over missingness × imputation × learner.

    Each run starts from the same deterministic train/test split
    and applies a new random mask per (mechanism, rate).
    """
    if rates is None:
        rates = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    rng = np.random.RandomState(seed)
    idx = np.arange(len(X_full))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    Xtr_base = X_full.iloc[train_idx].copy()
    Xte_base = X_full.iloc[test_idx].copy()
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

    logit = LogisticRegression(max_iter=200, solver="lbfgs")
    tree = HistGradientBoostingClassifier(max_depth=3, random_state=seed)
    nn = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=seed,
    )

    target_col = "income" if mechanism == "INCOME_MCAR" else "dti_numeric"

    records = []

    for it in range(n_iter):
        for rate in rates:
            # fresh copies
            Xtr = Xtr_base.copy()
            Xte = Xte_base.copy()

            miss_mask = apply_missingness(X_full, rate, mechanism, target_col, rng)
            miss_train, miss_test = miss_mask[train_idx], miss_mask[test_idx]
            Xtr.loc[miss_train, target_col] = np.nan
            Xte.loc[miss_test, target_col] = np.nan

            for imputer_name, imputer_fn in {
                "mean": impute_mean,
                "logit": impute_linear,
                "tree": impute_tree,
            }.items():
                for add_flag in [False, True]:
                    Xtr_imp = Xtr.copy()
                    Xte_imp = Xte.copy()
                    imputer_fn(Xtr_imp, Xte_imp, target_col)

                    if add_flag:
                        flag_name = f"{target_col}_missing"
                        Xtr_imp[flag_name] = Xtr[target_col].isna().astype(int)
                        Xte_imp[flag_name] = Xte[target_col].isna().astype(int)

                    for learner_name, learner in {
                        "Logit": logit,
                        "Tree": tree,
                        "NN": nn,
                    }.items():
                        scaler = StandardScaler().fit(Xtr_imp)
                        auc, r2, curv = eval_model(
                            learner,
                            scaler.transform(Xtr_imp),
                            ytr,
                            scaler.transform(Xte_imp),
                            yte,
                        )
                        records.append(
                            {
                                "iter": it,
                                "missing": rate,
                                "mechanism": mechanism,
                                "target": target_col,
                                "imputer": imputer_name,
                                "flag": add_flag,
                                "learner": learner_name,
                                "auc": auc,
                                "r2": r2,
                                "curv": curv,
                            }
                        )
    return pd.DataFrame(records)

# =========================================================
# Descriptive statistics
# =========================================================
def compute_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics with missingness percent."""
    desc = df.describe().T
    desc["missing_%"] = df.isna().mean() * 100
    desc = desc[["count", "mean", "std", "min", "50%", "max", "missing_%"]]
    desc.rename(columns={"50%": "median"}, inplace=True)
    return desc


# =========================================================
# Visualization
# =========================================================
def plot_clean_panels(df_results: pd.DataFrame, output_dir: Path):
    """Generate publication-ready panels for (imputer × mech × flag)."""
    palette = {"Logit": "#4477AA", "NN": "#66CCEE", "Tree": "#AA3377"}
    mechanisms = ["INCOME_MCAR", "MCAR", "MAR", "MNAR"]
    imputers = ["mean", "logit", "tree"]
    flags = [False, True]

    for imputer in imputers:
        for flag_state in flags:
            for mech in mechanisms:
                sub = df_results[
                    (df_results["imputer"] == imputer)
                    & (df_results["mechanism"] == mech)
                    & (df_results["flag"] == flag_state)
                ]
                if sub.empty:
                    continue

                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                sns.lineplot(
                    data=sub,
                    x="missing", y="auc",
                    hue="learner", palette=palette,
                    estimator="mean", ci="sd", marker="o",
                    ax=ax, legend=False
                )
                ax.set_ylim(0.62, 0.79)
                ax.set_xlabel("Proportion Missing")
                ax.set_ylabel("AUC")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                flag_label = "flag1" if flag_state else "flag0"
                fname = f"{imputer}_{mech}_{flag_label}.png"
                fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  saved {fname}")


def generate_legend(df_results: pd.DataFrame, output_dir: Path):
    """Generate standalone legend figure."""
    palette = {"Logit": "#4477AA", "NN": "#66CCEE", "Tree": "#AA3377"}
    sub = df_results[
        (df_results["imputer"] == "mean")
        & (df_results["mechanism"] == "MCAR")
        & (df_results["flag"] == False)
    ].head(10)

    fig, ax = plt.subplots(figsize=(4, 1))
    sns.lineplot(
        data=sub, x="missing", y="auc",
        hue="learner", palette=palette,
        estimator="mean", ci="sd", marker="o", ax=ax
    )
    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)

    fig_leg = plt.figure(figsize=(4.5, 0.5))
    fig_leg.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=9)
    plt.tight_layout()
    fig_leg.savefig(output_dir / "legend_horizontal.png", dpi=300, bbox_inches="tight")
    plt.close(fig_leg)
    print("  saved legend_horizontal.png")


# =========================================================
# Main execution
# =========================================================
def main():
    print("=" * 68)
    print(" HMDA Missingness & Imputation Experiment ")
    print("=" * 68)

    # -----------------------------------------------------
    # 1. Load and clean HMDA
    # -----------------------------------------------------
    print("[1/5] Loading HMDA data...")
    df_raw = load_hmda(DATA_PATH, chunk_size=CHUNK_SIZE)
    df_clean = clean_data(df_raw)

    # -----------------------------------------------------
    # 2. Descriptive statistics (Table 1)
    # -----------------------------------------------------
    print("[2/5] Computing full-sample descriptives...")
    table1_full = compute_descriptives(df_clean)
    table1_full.to_csv(RESULTS_DIR / "table1_fullsample.csv")
    print("  saved results/table1_fullsample.csv")

    # -----------------------------------------------------
    # 3. Monte Carlo draws
    # -----------------------------------------------------
    print(f"[3/5] Running {N_DRAWS} Monte Carlo draws...")
    all_records = []
    for draw in range(N_DRAWS):
        print(f"  → Draw {draw+1}/{N_DRAWS}")
        df_exp = df_clean.sample(100_000, random_state=SEED + draw)
        df_exp = df_exp.dropna(
            subset=["income", "dti_numeric", "age_numeric", "loan_amount", "region"]
        ).copy()

        region_dummies = pd.get_dummies(df_exp["region"], drop_first=True)
        X_full = pd.concat(
            [df_exp[["loan_amount", "dti_numeric", "age_numeric", "income"]], region_dummies],
            axis=1,
        )
        y = df_exp["denied"].astype(int)

        table1_draw = compute_descriptives(df_exp)
        table1_draw["draw"] = draw
        table1_draw.to_csv(RESULTS_DIR / f"table1_draw{draw}.csv")

        for mech in ["INCOME_MCAR", "MCAR", "MAR", "MNAR"]:
            df_mech = run_missingness_sweep(X_full, y, mechanism=mech, n_iter=1)
            df_mech["draw"] = draw
            all_records.append(df_mech)

    df_all = pd.concat(all_records, ignore_index=True)
    df_all.to_csv(RESULTS_DIR / "monte_outputs.csv", index=False)
    print("  saved results/monte_outputs.csv")

    # -----------------------------------------------------
    # 4. Plot panels
    # -----------------------------------------------------
    print("[4/5] Generating figures...")
    plot_clean_panels(df_all, FIGURES_DIR)
    generate_legend(df_all, FIGURES_DIR)

    # -----------------------------------------------------
    # 5. Done
    # -----------------------------------------------------
    print("=" * 68)
    print(" Experiment complete.")
    print(f" Results → {RESULTS_DIR}")
    print(f" Figures → {FIGURES_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()


# In[ ]:




