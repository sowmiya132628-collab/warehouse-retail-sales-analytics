"""
============================================================
Warehouse & Retail Sales Intelligence
4-Tier Data Analytics + Machine Learning Pipeline
============================================================
Author  : Sowmiya Subramaniyan
Dataset : Warehouse_and_Retail_Sales.csv
Target  : High_Warehouse_Demand (derived - see §8)
============================================================
"""

# -- 0. Imports --------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

# -- Paths -------------------------------------------------
DATA_PATH   = "Warehouse_and_Retail_Sales.csv"
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT    = "#58a6ff"
ACCENT2   = "#3fb950"
ACCENT3   = "#f78166"
TEXT_COL  = "#c9d1d9"
MUTED     = "#8b949e"

plt.rcParams.update({
    "figure.facecolor"  : DARK_BG,
    "axes.facecolor"    : CARD_BG,
    "axes.edgecolor"    : MUTED,
    "axes.labelcolor"   : TEXT_COL,
    "xtick.color"       : TEXT_COL,
    "ytick.color"       : TEXT_COL,
    "text.color"        : TEXT_COL,
    "grid.color"        : "#21262d",
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "legend.facecolor"  : CARD_BG,
    "legend.edgecolor"  : MUTED,
    "figure.dpi"        : 150,
})


# ----------------------------------------------------------
# TIER 1 - DATA HYGIENE & ARCHITECTURE
# ----------------------------------------------------------

def load_and_audit(path: str) -> tuple[pd.DataFrame, dict]:
    """Load CSV and return (raw_df, audit_dict)."""
    df = pd.read_csv(path)

    audit = {
        "shape"       : df.shape,
        "columns"     : df.columns.tolist(),
        "dtypes"      : df.dtypes.to_dict(),
        "missing"     : df.isnull().sum().to_dict(),
        "missing_pct" : (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates"  : int(df.duplicated().sum()),
        "nunique"     : df.nunique().to_dict(),
        "describe"    : df.describe().to_dict(),
        "neg_retail_sales"      : int((df["RETAIL SALES"] < 0).sum()),
        "neg_retail_transfers"  : int((df["RETAIL TRANSFERS"] < 0).sum()),
        "neg_warehouse_sales"   : int((df["WAREHOUSE SALES"] < 0).sum()),
        "year_dist"   : df["YEAR"].value_counts().sort_index().to_dict(),
        "item_type_dist": df["ITEM TYPE"].value_counts().to_dict(),
        "top_suppliers" : df["SUPPLIER"].value_counts().head(10).to_dict(),
    }

    print("=" * 60)
    print("DATA QUALITY AUDIT")
    print("=" * 60)
    print(f"  Rows          : {audit['shape'][0]:,}")
    print(f"  Columns       : {audit['shape'][1]}")
    print(f"  Columns       : {audit['columns']}")
    print(f"  Duplicates    : {audit['duplicates']}")
    print("\nMissing values:")
    for col, cnt in audit["missing"].items():
        if cnt:
            print(f"  {col:30s}: {cnt:6,}  ({audit['missing_pct'][col]:.2f}%)")
    print("\nNegative values:")
    print(f"  RETAIL SALES      : {audit['neg_retail_sales']:,}")
    print(f"  RETAIL TRANSFERS  : {audit['neg_retail_transfers']:,}")
    print(f"  WAREHOUSE SALES   : {audit['neg_warehouse_sales']:,}")
    print("=" * 60)
    return df, audit


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning log:
    ---------------------------------------------------------
    Problem  → Action                              → Reason
    ---------------------------------------------------------
    167 NULL SUPPLIER rows
             → Fill with 'UNKNOWN SUPPLIER'        → Preserve rows; unknown ≠ invalid sale
    1 NULL ITEM TYPE row
             → Fill with 'UNKNOWN'                 → Single row; removal not warranted
    3 NULL RETAIL SALES
             → Fill with 0.0                        → Absence of sale data treated as zero
    716 negative WAREHOUSE SALES
             → Clip to 0 (treat as returns/adj)    → Negative units indicate returns; floor at 0
    1016 negative RETAIL TRANSFERS
             → Clip to 0                            → Same rationale as above
    113 negative RETAIL SALES
             → Clip to 0                            → Same rationale as above
    ITEM CODE mixed int/str
             → Cast to str                          → Identifier; preserve leading zeros
    ---------------------------------------------------------
    """
    df = df.copy()

    # Fill missing categoricals
    df["SUPPLIER"]    = df["SUPPLIER"].fillna("UNKNOWN SUPPLIER")
    df["ITEM TYPE"]   = df["ITEM TYPE"].fillna("UNKNOWN")
    df["RETAIL SALES"] = df["RETAIL SALES"].fillna(0.0)

    # Clip negative values (returns / adjustments)
    df["RETAIL SALES"]      = df["RETAIL SALES"].clip(lower=0)
    df["RETAIL TRANSFERS"]  = df["RETAIL TRANSFERS"].clip(lower=0)
    df["WAREHOUSE SALES"]   = df["WAREHOUSE SALES"].clip(lower=0)

    # Normalise ITEM CODE to string
    df["ITEM CODE"] = df["ITEM CODE"].astype(str).str.strip()

    # Normalise string columns
    for col in ["SUPPLIER", "ITEM TYPE", "ITEM DESCRIPTION"]:
        df[col] = df[col].str.strip().str.upper()

    # Strip non-standard ITEM TYPEs into a single bucket for ML
    # (STR_SUPPLIES, REF, DUNNAGE represent < 0.2% of data each)
    minor_types = {"STR_SUPPLIES", "REF", "DUNNAGE"}
    df["ITEM TYPE CLEAN"] = df["ITEM TYPE"].apply(
        lambda x: "OTHER" if x in minor_types else x
    )

    # Derive DATE column for time-series
    df["DATE"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01"
    )

    # Derive TOTAL SALES (sum of retail + warehouse movement)
    df["TOTAL MOVEMENT"] = df["RETAIL SALES"] + df["RETAIL TRANSFERS"] + df["WAREHOUSE SALES"]

    return df


# ----------------------------------------------------------
# TIER 2 - EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------

def eda_viz1_monthly_trend(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Fig 1 - Monthly Warehouse & Retail Sales Trend (2017-2020)"""
    monthly = (
        df.groupby("DATE")[["WAREHOUSE SALES", "RETAIL SALES", "RETAIL TRANSFERS"]]
        .sum()
        .reset_index()
        .sort_values("DATE")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["DATE"], monthly["WAREHOUSE SALES"],
            color=ACCENT, lw=2, label="Warehouse Sales")
    ax.plot(monthly["DATE"], monthly["RETAIL SALES"],
            color=ACCENT2, lw=2, label="Retail Sales")
    ax.plot(monthly["DATE"], monthly["RETAIL TRANSFERS"],
            color=ACCENT3, lw=2, linestyle="--", label="Retail Transfers")

    ax.set_title("Monthly Sales Volume Trend (2017-2020)", fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold / Transferred")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig1_monthly_trend.png")
    return fig


def eda_viz2_item_type_dist(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Fig 2 - Warehouse Sales by Item Type"""
    type_sales = (
        df.groupby("ITEM TYPE CLEAN")["WAREHOUSE SALES"]
        .sum()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(type_sales.index, type_sales.values,
                   color=[ACCENT, ACCENT2, ACCENT3, "#d2a8ff", "#ffa657", "#79c0ff"],
                   edgecolor="none")
    for bar in bars:
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():,.0f}", va="center", fontsize=8, color=TEXT_COL)

    ax.set_title("Total Warehouse Sales Volume by Item Type", fontsize=14, pad=12)
    ax.set_xlabel("Total Warehouse Units Sold")
    ax.set_ylabel("Item Type")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig2_item_type.png")
    return fig


def eda_viz3_top_suppliers(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Fig 3 - Top 15 Suppliers by Total Warehouse Sales"""
    top_sup = (
        df.groupby("SUPPLIER")["WAREHOUSE SALES"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(top_sup.index, top_sup.values, color=ACCENT, edgecolor="none")
    for bar in bars:
        ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():,.0f}", va="center", fontsize=7.5, color=TEXT_COL)

    ax.set_title("Top 15 Suppliers - Total Warehouse Sales Volume", fontsize=14, pad=12)
    ax.set_xlabel("Total Warehouse Units Sold")
    ax.set_ylabel("Supplier")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig3_top_suppliers.png")
    return fig


def eda_viz4_correlation_scatter(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Fig 4 - Correlation Heatmap of Numeric Variables"""
    num_cols = ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES", "TOTAL MOVEMENT"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f",
        cmap="Blues", linewidths=0.5, linecolor=DARK_BG,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "color": "white"},
    )
    ax.set_title("Pearson Correlation - Numeric Sales Variables", fontsize=13, pad=10)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig4_correlation.png")
    return fig


def eda_viz5_heatmap_type_year(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Fig 5 - Warehouse Sales Heatmap: Item Type × Year"""
    pivot = (
        df.groupby(["YEAR", "ITEM TYPE CLEAN"])["WAREHOUSE SALES"]
        .sum()
        .unstack("ITEM TYPE CLEAN")
        .fillna(0)
    )
    # Normalise each column for visual comparability
    pivot_norm = pivot.div(pivot.max())

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot_norm, ax=ax, cmap="YlOrRd", annot=False,
        linewidths=0.5, linecolor=DARK_BG,
        cbar_kws={"label": "Relative Volume (0-1)"},
    )
    ax.set_title("Warehouse Sales Intensity: Item Type × Year (Normalised)", fontsize=13, pad=10)
    ax.set_xlabel("Item Type")
    ax.set_ylabel("Year")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "fig5_type_year_heatmap.png")
    return fig


def run_eda(df: pd.DataFrame) -> None:
    """Generate all 5 EDA visualisations."""
    print("\n[EDA] Generating visualisations …")
    eda_viz1_monthly_trend(df)
    eda_viz2_item_type_dist(df)
    eda_viz3_top_suppliers(df)
    eda_viz4_correlation_scatter(df)
    eda_viz5_heatmap_type_year(df)
    print(f"[EDA] Figures saved to {FIGURES_DIR}/")


# ----------------------------------------------------------
# TIER 3 - PREDICTIVE MACHINE LEARNING
# ----------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    -- §8  TARGET DEFINITION ------------------------------
    No explicit binary target exists in the source system.
    Derived target: High_Warehouse_Demand

    Construction:
      • Aggregate WAREHOUSE SALES at (YEAR, MONTH, ITEM CODE) grain
        (already the grain of this dataset)
      • High_Warehouse_Demand = 1  if WAREHOUSE SALES ≥ 90th percentile
                               0  otherwise
    Threshold (p90) makes ~10% of records "high demand" -
    operationally meaningful as the items requiring priority
    stock replenishment attention.
    This is a DERIVED ANALYTICAL TARGET; it does not exist
    in the source system and is documented as such.

    -- §9  LEAKAGE PREVENTION ----------------------------
    Excluded features:
      WAREHOUSE SALES   - IS the target; direct leak
      TOTAL MOVEMENT    - derived from WAREHOUSE SALES; indirect leak
      RETAIL SALES      - contemporaneous outcome; post-observation
                          variable not known before the period closes
      RETAIL TRANSFERS  - same timing issue as RETAIL SALES
      ITEM DESCRIPTION  - near-perfect proxy for ITEM CODE (identifier)
      ITEM CODE         - raw high-cardinality identifier; would memorise
                          not generalise; removed
      DATE              - information already captured by YEAR + MONTH

    Kept features:
      YEAR, MONTH, ITEM TYPE CLEAN, SUPPLIER (encoded)
    ------------------------------------------------------
    """
    feat = df.copy()

    # Derive target
    p90 = feat["WAREHOUSE SALES"].quantile(0.90)
    feat["High_Warehouse_Demand"] = (feat["WAREHOUSE SALES"] >= p90).astype(int)

    # Encode categoricals
    le_type = LabelEncoder()
    le_sup  = LabelEncoder()
    feat["ITEM_TYPE_ENC"] = le_type.fit_transform(feat["ITEM TYPE CLEAN"])
    feat["SUPPLIER_ENC"]  = le_sup.fit_transform(feat["SUPPLIER"])

    # Feature matrix - leakage-free
    FEATURES = ["YEAR", "MONTH", "ITEM_TYPE_ENC", "SUPPLIER_ENC"]
    TARGET   = "High_Warehouse_Demand"

    return feat, FEATURES, TARGET, p90, le_type, le_sup


def train_model(df: pd.DataFrame) -> dict:
    """Train model and return results dict."""
    print("\n[ML] Building feature matrix …")
    feat, FEATURES, TARGET, p90, le_type, le_sup = build_features(df)

    X = feat[FEATURES]
    y = feat[TARGET]

    # Chronological split: 2017-2019 = train, 2020 = test
    # Rationale: avoids leaking future period information into training.
    # A random split would mix 2020 observations into training and
    # create temporal leakage.
    train_mask = feat["YEAR"] < 2020
    test_mask  = feat["YEAR"] == 2020

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"[ML] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"[ML] Target positive rate (train): {y_train.mean():.3f}")

    # Model: Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred      = clf.predict(X_test)
    y_prob      = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc"   : round(roc_auc_score(y_test, y_prob), 4),
    }

    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature"   : FEATURES,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Prediction table (test set)
    pred_df = feat[test_mask].copy()
    pred_df["Prob_High_Demand"] = y_prob
    pred_df["Predicted_Class"]  = y_pred
    pred_df["Actual_Class"]     = y_test.values

    print("\n[ML] -- Model Results ------------------------------------------")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("-" * 52)

    return {
        "model"       : clf,
        "metrics"     : metrics,
        "cm"          : cm,
        "fpr"         : fpr,
        "tpr"         : tpr,
        "importance"  : importance_df,
        "pred_df"     : pred_df,
        "p90_thresh"  : p90,
        "X_train"     : X_train,
        "X_test"      : X_test,
        "y_train"     : y_train,
        "y_test"      : y_test,
        "y_prob"      : y_prob,
        "features"    : FEATURES,
        "le_type"     : le_type,
        "le_sup"      : le_sup,
    }


def plot_model_results(results: dict) -> None:
    """Save confusion matrix and ROC curve figures."""
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=results["cm"],
        display_labels=["Normal", "High Demand"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix - Random Forest", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cm.png")
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(results["fpr"], results["tpr"], color=ACCENT, lw=2,
            label=f"ROC-AUC = {results['metrics']['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - High Warehouse Demand Classifier", fontsize=13, pad=10)
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_roc.png")
    plt.close(fig)

    # Feature importance
    fig, ax = plt.subplots(figsize=(8, 4))
    imp = results["importance"].sort_values("Importance")
    ax.barh(imp["Feature"], imp["Importance"], color=ACCENT, edgecolor="none")
    ax.set_title("Feature Importance - Random Forest", fontsize=13, pad=10)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_importance.png")
    plt.close(fig)

    print(f"[ML] Model figures saved to {FIGURES_DIR}/")


# ----------------------------------------------------------
# TIER 4 - PRESCRIPTIVE ANALYTICS
# ----------------------------------------------------------

def prescriptive_priority_queue(pred_df: pd.DataFrame, capacity: int = 500) -> pd.DataFrame:
    """
    Resource-Constrained Priority Engine
    --------------------------------------
    The inventory team has limited bandwidth to manually review
    and pre-order high-demand stock each planning cycle.

    Operational rule:
      Rank all item-month combinations by predicted high-demand
      probability (descending). Return only the top `capacity`
      records for manual review and pre-stocking action.

    Capacity is configurable in the dashboard.
    """
    queue = (
        pred_df[["YEAR", "MONTH", "SUPPLIER", "ITEM CODE",
                 "ITEM DESCRIPTION", "ITEM TYPE",
                 "Prob_High_Demand", "Predicted_Class", "WAREHOUSE SALES"]]
        .copy()
        .sort_values("Prob_High_Demand", ascending=False)
        .reset_index(drop=True)
    )
    queue.index += 1
    queue.index.name = "Rank"

    queue["Recommended Action"] = queue["Predicted_Class"].map({
        1: "🔴 PRE-STOCK - Urgent replenishment recommended",
        0: "🟢 MONITOR   - Normal; review if capacity allows",
    })

    return queue.head(capacity)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def run_pipeline() -> dict:
    """Execute the full 4-tier pipeline and return artefacts."""
    # Tier 1
    raw_df, audit  = load_and_audit(DATA_PATH)
    clean_df       = clean_data(raw_df)

    # Tier 2
    run_eda(clean_df)

    # Tier 3
    results        = train_model(clean_df)
    plot_model_results(results)

    # Tier 4
    priority_queue = prescriptive_priority_queue(results["pred_df"], capacity=500)

    print("\n[PRESCRIPTIVE] Top-5 Priority Queue:")
    print(priority_queue[["SUPPLIER", "ITEM TYPE",
                           "Prob_High_Demand", "Recommended Action"]].head(5).to_string())

    artefacts = {
        "raw_df"        : raw_df,
        "clean_df"      : clean_df,
        "audit"         : audit,
        "results"       : results,
        "priority_queue": priority_queue,
    }
    return artefacts


if __name__ == "__main__":
    run_pipeline()
    print("\n✅ Pipeline complete. Check figures/ directory for charts.")
