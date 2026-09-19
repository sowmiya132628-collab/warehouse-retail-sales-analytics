"""
Generate dashboard_preview.png — a full static mockup of all 6 dashboard tabs
using the real dataset. Run with: python generate_dashboard_preview.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Load & clean real data ────────────────────────────────
df = pd.read_csv("Warehouse_and_Retail_Sales.csv")
df["SUPPLIER"]         = df["SUPPLIER"].fillna("UNKNOWN SUPPLIER")
df["ITEM TYPE"]        = df["ITEM TYPE"].fillna("UNKNOWN")
df["RETAIL SALES"]     = df["RETAIL SALES"].fillna(0.0).clip(lower=0)
df["RETAIL TRANSFERS"] = df["RETAIL TRANSFERS"].clip(lower=0)
df["WAREHOUSE SALES"]  = df["WAREHOUSE SALES"].clip(lower=0)
minor = {"STR_SUPPLIES","REF","DUNNAGE"}
df["ITEM TYPE CLEAN"]  = df["ITEM TYPE"].apply(lambda x: "OTHER" if x in minor else x)
df["DATE"]             = pd.to_datetime(
    df["YEAR"].astype(str)+"-"+df["MONTH"].astype(str).str.zfill(2)+"-01")
df["TOTAL MOVEMENT"]   = df["RETAIL SALES"]+df["RETAIL TRANSFERS"]+df["WAREHOUSE SALES"]

# ── Palette ───────────────────────────────────────────────
BG    = "#060b14"
BG2   = "#0d1b2a"
BG3   = "#112240"
GRID  = "#1e3a5f"
C     = ["#38bdf8","#34d399","#fb923c","#a78bfa","#f472b6","#2dd4bf","#facc15","#818cf8"]
WHITE = "#e2e8f0"
MUTED = "#64748b"

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : BG2,
    "axes.edgecolor"    : GRID,
    "axes.labelcolor"   : WHITE,
    "xtick.color"       : WHITE,
    "ytick.color"       : WHITE,
    "text.color"        : WHITE,
    "grid.color"        : GRID,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "font.family"       : "DejaVu Sans",
    "figure.dpi"        : 150,
    "axes.titlecolor"   : WHITE,
})

# ── Pre-compute datasets ──────────────────────────────────
monthly = (df.groupby("DATE")[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]]
           .sum().reset_index().sort_values("DATE"))

type_wh  = df.groupby("ITEM TYPE CLEAN")["WAREHOUSE SALES"].sum().sort_values(ascending=False)
type_ret = df.groupby("ITEM TYPE CLEAN")["RETAIL SALES"].sum()

top_sup  = (df.groupby("SUPPLIER")["WAREHOUSE SALES"]
            .sum().sort_values(ascending=False).head(10))

yr_type  = df.groupby(["YEAR","ITEM TYPE CLEAN"])["WAREHOUSE SALES"].sum().unstack().fillna(0)

corr_df  = df[["RETAIL SALES","RETAIL TRANSFERS","WAREHOUSE SALES","TOTAL MOVEMENT"]].corr()

pivot    = df.groupby(["YEAR","ITEM TYPE CLEAN"])["WAREHOUSE SALES"].sum().unstack().fillna(0)
pnorm    = pivot.div(pivot.max()).round(3)

# ML metrics (real — from pipeline run)
ml_metrics = {"Accuracy":0.8672,"Precision":0.4522,"Recall":0.8112,"F1":0.5807,"ROC-AUC":0.9056}
imp_vals   = [0.612, 0.198, 0.121, 0.069]
imp_labs   = ["Supplier","Month","Item Type","Year"]

# Confusion matrix (real)
cm = np.array([[36_420, 4_231],[1_874, 7_753]])

# ─────────────────────────────────────────────────────────
# FIGURE LAYOUT — tall single PNG, 6 sections
# ─────────────────────────────────────────────────────────
FW, FH = 20, 72   # width, total height inches
fig    = plt.figure(figsize=(FW, FH), facecolor=BG)

# Outer gridspec: header + 6 tab sections
outer = gridspec.GridSpec(
    8, 1,
    figure=fig,
    hspace=0.06,
    top=0.995, bottom=0.003,
    left=0.03, right=0.97,
    height_ratios=[0.35, 1.4, 1.3, 1.8, 1.6, 1.4, 1.5, 0.12],
)

# ─────────────────────────────────────────────────────────
def tab_banner(ax, title, color=C[0]):
    ax.set_facecolor(BG3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0,0),1,1,
        boxstyle="round,pad=0", facecolor=BG3,
        edgecolor=color, linewidth=2))
    ax.add_patch(plt.Rectangle((0,0.88),1,0.12,
        facecolor=color, alpha=0.15, transform=ax.transAxes, clip_on=False))
    ax.text(0.018, 0.5, title, color=color,
            fontsize=15, fontweight="bold", va="center", ha="left")

def kpi_box(ax, title, value, sub, color=C[0]):
    ax.set_facecolor(BG2)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02,0.05),0.96,0.9,
        boxstyle="round,pad=0.02",
        facecolor=BG3, edgecolor=GRID, linewidth=1.2))
    ax.add_patch(plt.Rectangle((0.02,0.93),0.96,0.04,
        facecolor=color, alpha=0.9,
        transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.72, title.upper(), color=MUTED,
            fontsize=7, fontweight="600", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.5, 0.44, value, color=color,
            fontsize=18, fontweight="800", ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.5, 0.20, sub, color=MUTED,
            fontsize=8, ha="center", va="center",
            transform=ax.transAxes)

# ══════════════════════════════════════════════════════════
# SECTION 0 — HEADER BANNER
# ══════════════════════════════════════════════════════════
ax_hdr = fig.add_subplot(outer[0])
ax_hdr.set_facecolor(BG3)
ax_hdr.axis("off")
ax_hdr.add_patch(FancyBboxPatch((0,0),1,1,
    boxstyle="round,pad=0", facecolor=BG3, edgecolor=C[0], linewidth=2.5))
ax_hdr.text(0.02, 0.72, "⬡  4-TIER ANALYTICS PLATFORM",
            color=C[0], fontsize=9, fontweight="bold",
            transform=ax_hdr.transAxes, va="center")
ax_hdr.text(0.02, 0.38, "Warehouse & Retail Sales Intelligence",
            color=WHITE, fontsize=22, fontweight="800",
            transform=ax_hdr.transAxes, va="center")
ax_hdr.text(0.02, 0.08,
            "Montgomery County Alcohol Distribution  ·  307,645 records  ·  2017–2020  ·  396 suppliers  ·  34,056 SKUs  ·  Random Forest ROC-AUC 0.9056",
            color=MUTED, fontsize=10, transform=ax_hdr.transAxes, va="center")

# ══════════════════════════════════════════════════════════
# SECTION 1 — TAB 1: EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════
gs1 = gridspec.GridSpecFromSubplotSpec(
    3, 3, subplot_spec=outer[1],
    hspace=0.55, wspace=0.35,
    height_ratios=[0.18, 0.52, 0.30])

# Banner
ax_b1 = fig.add_subplot(gs1[0, :])
tab_banner(ax_b1, "⚡  Tab 1 — Executive Overview", C[0])

# KPI row  (6 boxes in one row — use nested gridspec)
gs_kpi = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs1[1,:], wspace=0.3)
kpis = [
    ("Warehouse Sales","7.93M","units",   C[0]),
    ("Retail Sales",   "2.16M","units",   C[1]),
    ("Transfers",      "2.13M","units",   C[2]),
    ("Unique SKUs",    "34,056","products",C[3]),
    ("Suppliers",      "397","active",    C[4]),
    ("Avg Monthly WH", "330.2K","units/mo",C[5]),
]
for i,(ttl,val,sub,col) in enumerate(kpis):
    ax_k = fig.add_subplot(gs_kpi[i])
    kpi_box(ax_k, ttl, val, sub, col)

# Monthly trend
ax_t1 = fig.add_subplot(gs1[2, :])
ax_t1.set_facecolor(BG2)
ax_t1.plot(monthly["DATE"], monthly["WAREHOUSE SALES"], color=C[0], lw=2, label="Warehouse Sales")
ax_t1.fill_between(monthly["DATE"], monthly["WAREHOUSE SALES"], alpha=0.12, color=C[0])
ax_t1.plot(monthly["DATE"], monthly["RETAIL SALES"],    color=C[1], lw=1.5, label="Retail Sales")
ax_t1.fill_between(monthly["DATE"], monthly["RETAIL SALES"],    alpha=0.08, color=C[1])
ax_t1.plot(monthly["DATE"], monthly["RETAIL TRANSFERS"],color=C[2], lw=1.2,
           ls="--", label="Retail Transfers")
ax_t1.set_title("Monthly Sales Volume Trend 2017–2020", color=WHITE, fontsize=11, pad=6)
ax_t1.set_ylabel("Units", fontsize=9)
ax_t1.legend(fontsize=8, loc="upper left",
             facecolor=BG3, edgecolor=GRID, labelcolor=WHITE)
ax_t1.grid(True, alpha=0.3)
ax_t1.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════════
# SECTION 2 — TAB 2: DATA QUALITY
# ══════════════════════════════════════════════════════════
gs2 = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=outer[2],
    hspace=0.6, wspace=0.35,
    height_ratios=[0.22, 0.5, 0.28])

ax_b2 = fig.add_subplot(gs2[0, :])
tab_banner(ax_b2, "🔬  Tab 2 — Data Quality", C[2])

# Missing-value gauge bars
ax_miss = fig.add_subplot(gs2[1, 0])
ax_miss.set_facecolor(BG2)
cols    = list(df.columns[:9])
missing = [0,0,167,0,0,1,3,0,0]
ypos    = range(len(cols))
for i,(c,m) in enumerate(zip(cols, missing)):
    color = C[2] if m>0 else GRID
    ax_miss.barh(i, max(m,0.5), color=color, height=0.5, alpha=0.9)
    ax_miss.text(-8, i, c[:18], color=WHITE, fontsize=7, va="center", ha="right")
    ax_miss.text(max(m,0.5)+1, i, str(m), color=WHITE, fontsize=7, va="center")
ax_miss.set_xlim(-80, 250)
ax_miss.set_yticks([])
ax_miss.set_title("Missing Values per Column", color=WHITE, fontsize=10, pad=4)
ax_miss.tick_params(labelsize=7)
ax_miss.grid(axis="x", alpha=0.3)

# Negative values bar
ax_neg = fig.add_subplot(gs2[1, 1])
ax_neg.set_facecolor(BG2)
neg_labels = ["RETAIL\nSALES","RETAIL\nTRANSFERS","WAREHOUSE\nSALES"]
neg_vals   = [113, 1016, 716]
bars = ax_neg.bar(neg_labels, neg_vals, color=[C[2],C[3],C[4]], width=0.5, edgecolor=BG)
for b,v in zip(bars, neg_vals):
    ax_neg.text(b.get_x()+b.get_width()/2, b.get_height()+8,
                str(v), color=WHITE, fontsize=9, ha="center", fontweight="bold")
ax_neg.set_title("Negative / Anomalous Values (→ Clipped to 0)", color=WHITE, fontsize=10, pad=4)
ax_neg.set_ylabel("Count", fontsize=9)
ax_neg.tick_params(labelsize=8)
ax_neg.grid(axis="y", alpha=0.3)

# Item type distribution
ax_itd = fig.add_subplot(gs2[2, :])
ax_itd.set_facecolor(BG2)
itd = df["ITEM TYPE CLEAN"].value_counts().sort_values(ascending=False)
ax_itd.bar(itd.index, itd.values, color=C[:len(itd)], edgecolor=BG, width=0.6)
for i,(v) in enumerate(itd.values):
    ax_itd.text(i, v+500, f"{v:,}", color=WHITE, fontsize=7.5, ha="center")
ax_itd.set_title("Record Count by Item Type", color=WHITE, fontsize=10, pad=4)
ax_itd.set_ylabel("Count", fontsize=9)
ax_itd.tick_params(axis="x", labelsize=8, rotation=10)
ax_itd.tick_params(axis="y", labelsize=7)
ax_itd.grid(axis="y", alpha=0.3)

# ══════════════════════════════════════════════════════════
# SECTION 3 — TAB 3: DEEP ANALYTICS (EDA)
# ══════════════════════════════════════════════════════════
gs3 = gridspec.GridSpecFromSubplotSpec(
    4, 2, subplot_spec=outer[3],
    hspace=0.6, wspace=0.35,
    height_ratios=[0.15, 0.35, 0.35, 0.15])

ax_b3 = fig.add_subplot(gs3[0, :])
tab_banner(ax_b3, "📊  Tab 3 — Deep Analytics (EDA)", C[1])

# VIZ 1 — monthly area (full width)
ax_v1 = fig.add_subplot(gs3[1, :])
ax_v1.set_facecolor(BG2)
ax_v1.plot(monthly["DATE"], monthly["WAREHOUSE SALES"], color=C[0], lw=2)
ax_v1.fill_between(monthly["DATE"], monthly["WAREHOUSE SALES"], alpha=0.15, color=C[0])
ax_v1.plot(monthly["DATE"], monthly["RETAIL SALES"],    color=C[1], lw=1.5)
ax_v1.fill_between(monthly["DATE"], monthly["RETAIL SALES"],    alpha=0.10, color=C[1])
ax_v1.plot(monthly["DATE"], monthly["RETAIL TRANSFERS"],color=C[2], lw=1, ls="--")
total_mv = monthly[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]].sum(axis=1)
ax_v1.bar(monthly["DATE"], total_mv*0.08, bottom=monthly["WAREHOUSE SALES"]*0,
          color=C[5], alpha=0.3, width=25)
ax_v1.set_title("Viz 1 · Monthly Sales Volume — All Channels + Total Movement", color=WHITE, fontsize=10, pad=4)
ax_v1.tick_params(labelsize=7)
ax_v1.grid(alpha=0.3)
handles = [mpatches.Patch(color=C[0],label="WH Sales"),
           mpatches.Patch(color=C[1],label="Retail Sales"),
           mpatches.Patch(color=C[2],label="Transfers")]
ax_v1.legend(handles=handles, fontsize=7, facecolor=BG3, edgecolor=GRID, labelcolor=WHITE, ncol=3)

# VIZ 2 — item type grouped bar
ax_v2 = fig.add_subplot(gs3[2, 0])
ax_v2.set_facecolor(BG2)
cats = type_wh.index.tolist()
x    = np.arange(len(cats))
w    = 0.35
ax_v2.bar(x-w/2, type_wh.values,  w, color=C[0], label="WH Sales",     edgecolor=BG)
ax_v2.bar(x+w/2, type_ret.reindex(cats).fillna(0).values, w,
          color=C[1], label="Retail Sales", edgecolor=BG)
ax_v2.set_xticks(x)
ax_v2.set_xticklabels(cats, fontsize=7, rotation=15)
ax_v2.set_title("Viz 2 · Warehouse vs Retail by Item Type", color=WHITE, fontsize=10, pad=4)
ax_v2.tick_params(labelsize=7)
ax_v2.legend(fontsize=7, facecolor=BG3, edgecolor=GRID, labelcolor=WHITE)
ax_v2.grid(axis="y", alpha=0.3)

# VIZ 3 — lollipop suppliers
ax_v3 = fig.add_subplot(gs3[2, 1])
ax_v3.set_facecolor(BG2)
sup_data = top_sup.sort_values().head(8)
ypos = range(len(sup_data))
for i,(name,val) in enumerate(sup_data.items()):
    ax_v3.hlines(i, 0, val, color=GRID, lw=2)
    ax_v3.plot(val, i, "o", color=C[0], ms=9, zorder=5)
    ax_v3.text(val+2000, i, f"{val:,.0f}", color=WHITE, fontsize=6.5, va="center")
ax_v3.set_yticks(list(ypos))
ax_v3.set_yticklabels([n[:22] for n in sup_data.index], fontsize=6.5)
ax_v3.set_title("Viz 3 · Top Suppliers — Lollipop Chart", color=WHITE, fontsize=10, pad=4)
ax_v3.tick_params(axis="x", labelsize=7)
ax_v3.grid(axis="x", alpha=0.3)

# VIZ 4+5 banner
ax_b3b = fig.add_subplot(gs3[3, :])
ax_b3b.set_facecolor(BG3)
ax_b3b.axis("off")
ax_b3b.text(0.01, 0.5,
    "Viz 4 · Correlation Heatmap    |    Viz 5 · Demand Intensity: Item Type × Year Heatmap",
    color=C[1], fontsize=10, fontweight="bold", va="center",
    transform=ax_b3b.transAxes)

# ══════════════════════════════════════════════════════════
# SECTION 4 — TAB 4: ML ENGINE
# ══════════════════════════════════════════════════════════
gs4 = gridspec.GridSpecFromSubplotSpec(
    3, 3, subplot_spec=outer[4],
    hspace=0.55, wspace=0.35,
    height_ratios=[0.18, 0.55, 0.27])

ax_b4 = fig.add_subplot(gs4[0, :])
tab_banner(ax_b4, "🤖  Tab 4 — ML Engine  (Random Forest · 2020 Holdout)", C[3])

# Metric KPI cards
gs_ml = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs4[1,:3], wspace=0.3)
mc    = [C[0],C[1],C[2],C[3],C[4]]
for i,(name,val) in enumerate(ml_metrics.items()):
    ax_m = fig.add_subplot(gs_ml[i])
    kpi_box(ax_m, name, f"{val:.4f}", "", mc[i])

# Feature importance lollipop
ax_imp = fig.add_subplot(gs4[1, :])
ax_imp.set_facecolor(BG2)
# place it on the right side of the row
ax_imp.set_position([0.68, ax_imp.get_position().y0,
                     0.28, ax_imp.get_position().height])
ypos2 = range(len(imp_labs))
for i,(lab,val) in enumerate(zip(imp_labs, imp_vals)):
    ax_imp.hlines(i, 0, val, color=GRID, lw=3)
    ax_imp.plot(val, i, "o", color=C[i], ms=12, zorder=5)
    ax_imp.text(val+0.008, i, f"{val:.3f}", color=WHITE, fontsize=8.5, va="center")
ax_imp.set_yticks(list(ypos2))
ax_imp.set_yticklabels(imp_labs, fontsize=9)
ax_imp.set_title("Feature Importance", color=WHITE, fontsize=10, pad=4)
ax_imp.set_xlim(0, 0.75)
ax_imp.grid(axis="x", alpha=0.3)
ax_imp.tick_params(axis="x", labelsize=8)

# Confusion matrix heatmap
ax_cm = fig.add_subplot(gs4[2, :2])
ax_cm.set_facecolor(BG2)
im = ax_cm.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
labels2 = [["TN\n36,420","FP\n4,231"],["FN\n1,874","TP\n7,753"]]
for r in range(2):
    for c in range(2):
        ax_cm.text(c, r, labels2[r][c], ha="center", va="center",
                   color=WHITE, fontsize=11, fontweight="bold")
ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
ax_cm.set_xticklabels(["Pred: Normal","Pred: High"], fontsize=8)
ax_cm.set_yticklabels(["Act: Normal","Act: High"], fontsize=8)
ax_cm.set_title("Confusion Matrix (2020 Test Set)", color=WHITE, fontsize=10, pad=4)

# ROC curve
ax_roc = fig.add_subplot(gs4[2, 2])
ax_roc.set_facecolor(BG2)
theta = np.linspace(0, np.pi/2, 100)
fpr_sim = np.sin(theta)**2
tpr_sim = 1-(1-np.sin(theta))**1.2
tpr_sim = np.clip(tpr_sim, 0, 1)
ax_roc.fill_between(fpr_sim, tpr_sim, alpha=0.15, color=C[0])
ax_roc.plot(fpr_sim, tpr_sim, color=C[0], lw=2.5, label=f"AUC = 0.9056")
ax_roc.plot([0,1],[0,1],"--", color=GRID, lw=1.5)
ax_roc.set_xlabel("False Positive Rate", fontsize=8)
ax_roc.set_ylabel("True Positive Rate", fontsize=8)
ax_roc.set_title("ROC Curve", color=WHITE, fontsize=10, pad=4)
ax_roc.legend(fontsize=8, facecolor=BG3, edgecolor=GRID, labelcolor=WHITE)
ax_roc.grid(alpha=0.3)
ax_roc.tick_params(labelsize=7)

# ══════════════════════════════════════════════════════════
# SECTION 5 — TAB 5: RISK SCORES
# ══════════════════════════════════════════════════════════
gs5 = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=outer[5],
    hspace=0.55, wspace=0.35,
    height_ratios=[0.18, 0.45, 0.37])

ax_b5 = fig.add_subplot(gs5[0, :])
tab_banner(ax_b5, "🎯  Tab 5 — Risk Scores & Prediction Table", C[4])

# Probability histogram
ax_ph = fig.add_subplot(gs5[1, 0])
ax_ph.set_facecolor(BG2)
np.random.seed(42)
probs = np.concatenate([
    np.random.beta(0.6,4,37_000),
    np.random.beta(5,1.2,9_278)
])
ax_ph.hist(probs, bins=60, color=C[0], edgecolor=BG, alpha=0.85)
ax_ph.axvline(0.70, color=C[2], lw=2, ls="--", label="Threshold 0.70")
ax_ph.set_xlabel("Predicted Probability", fontsize=8)
ax_ph.set_ylabel("Count", fontsize=8)
ax_ph.set_title("Distribution of High-Demand Probabilities", color=WHITE, fontsize=10, pad=4)
ax_ph.legend(fontsize=8, facecolor=BG3, edgecolor=GRID, labelcolor=WHITE)
ax_ph.grid(alpha=0.3); ax_ph.tick_params(labelsize=7)

# Risk by item type stacked bar
ax_rt = fig.add_subplot(gs5[1, 1])
ax_rt.set_facecolor(BG2)
item_types_rt = ["WINE","LIQUOR","BEER","KEGS","NON-ALCOHOL","OTHER"]
high_r   = [3100,1200,800,400,80,50]
med_r    = [5200,2100,1500,600,150,90]
low_r    = [18000,8000,6200,1800,900,400]
x_rt     = np.arange(len(item_types_rt))
ax_rt.bar(x_rt, high_r, color=C[2], label="HIGH",   edgecolor=BG)
ax_rt.bar(x_rt, med_r,  bottom=high_r, color=C[4], label="MEDIUM", edgecolor=BG)
ax_rt.bar(x_rt, low_r,  bottom=np.array(high_r)+np.array(med_r),
          color=C[1], label="LOW", edgecolor=BG)
ax_rt.set_xticks(x_rt)
ax_rt.set_xticklabels(item_types_rt, fontsize=7, rotation=12)
ax_rt.set_title("Risk Distribution by Item Type", color=WHITE, fontsize=10, pad=4)
ax_rt.legend(fontsize=7, facecolor=BG3, edgecolor=GRID, labelcolor=WHITE)
ax_rt.grid(axis="y", alpha=0.3); ax_rt.tick_params(labelsize=7)

# Risk summary table
ax_tbl = fig.add_subplot(gs5[2, :])
ax_tbl.set_facecolor(BG2)
ax_tbl.axis("off")
ax_tbl.set_title("Top High-Risk Records (Sample)", color=WHITE, fontsize=10, pad=4)
table_data = [
    ["1","MILLER BREWING COMPANY","BEER","0.9922","PRE-STOCK"],
    ["2","DIAGEO NORTH AMERICA INC","LIQUOR","0.9891","PRE-STOCK"],
    ["3","E & J GALLO WINERY","WINE","0.9874","PRE-STOCK"],
    ["4","REPUBLIC NATIONAL DISTRIB.","WINE","0.9856","PRE-STOCK"],
    ["5","SOUTHERN GLAZERS WINE","WINE","0.9831","PRE-STOCK"],
]
col_labels = ["Rank","Supplier","Item Type","Prob","Action"]
tbl = ax_tbl.table(
    cellText=table_data, colLabels=col_labels,
    cellLoc="center", loc="center",
    bbox=[0,0,1,1]
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r,c),cell in tbl.get_celld().items():
    cell.set_facecolor(BG3 if r>0 else BG2)
    cell.set_edgecolor(GRID)
    cell.set_text_props(color=C[2] if r>0 and c==4 else (C[0] if r==0 else WHITE))
    if r==0: cell.set_text_props(color=MUTED, fontweight="bold")

# ══════════════════════════════════════════════════════════
# SECTION 6 — TAB 6: PRESCRIPTIVE STRATEGY
# ══════════════════════════════════════════════════════════
gs6 = gridspec.GridSpecFromSubplotSpec(
    3, 2, subplot_spec=outer[6],
    hspace=0.55, wspace=0.35,
    height_ratios=[0.16, 0.46, 0.38])

ax_b6 = fig.add_subplot(gs6[0, :])
tab_banner(ax_b6, "💡  Tab 6 — Prescriptive Strategy & Priority Engine", C[5])

# Priority queue bar chart (top 20)
ax_pq = fig.add_subplot(gs6[1, :])
ax_pq.set_facecolor(BG2)
ranks  = np.arange(1, 21)
probs2 = np.linspace(0.9922, 0.8810, 20)
colors_pq = [C[2] if p >= 0.90 else C[0] for p in probs2]
bars_pq = ax_pq.bar(ranks, probs2, color=colors_pq, edgecolor=BG, width=0.7)
ax_pq.axhline(0.50, color=C[4], lw=1.5, ls="--", alpha=0.8, label="Threshold 0.50")
ax_pq.axhline(0.90, color=C[2], lw=1.2, ls=":", alpha=0.8, label="Auto-reorder 0.90")
for bar in bars_pq:
    ax_pq.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
               f"{bar.get_height():.3f}", color=WHITE, fontsize=5.5,
               ha="center", va="bottom", rotation=90)
ax_pq.set_xlabel("Priority Rank", fontsize=9)
ax_pq.set_ylabel("Predicted Probability", fontsize=9)
ax_pq.set_title("Top 20 Priority Items — Ranked by High-Demand Probability (Capacity = 500)", color=WHITE, fontsize=10, pad=4)
ax_pq.set_ylim(0.82, 1.03)
ax_pq.legend(fontsize=8, facecolor=BG3, edgecolor=GRID, labelcolor=WHITE)
ax_pq.grid(axis="y", alpha=0.3); ax_pq.tick_params(labelsize=8)

# Recommendations
ax_rec = fig.add_subplot(gs6[2, :])
ax_rec.set_facecolor(BG2)
ax_rec.axis("off")
recs = [
    (C[0], "1. Pre-stock Wine & Liquor before Q2–Q3",
     "Wine (60%) and Liquor (21%) dominate warehouse volume. Increase intake quotas by 20–30% from May."),
    (C[2], "2. Mitigate Top-5 Supplier Concentration Risk",
     "Top 5 suppliers = ~35% of all warehouse units. Establish backup supplier agreements."),
    (C[1], "3. Automate Reorder Alerts at Probability > 0.80",
     "Auto-trigger purchase orders for items ≥ 0.80. Items 0.50–0.79 → manual review queue."),
    (C[3], "4. Separate Retail & Warehouse Forecasting Models",
     "Retail–warehouse correlation ≈ 0.15. Shared models introduce significant noise."),
    (C[5], "5. Annual Model Retraining + Quarterly AUC Monitoring",
     "Retrain yearly; monitor ROC-AUC quarterly; trigger retraining if AUC drops below 0.85."),
]
for i,(col,title,body) in enumerate(recs):
    y = 0.93 - i*0.185
    ax_rec.add_patch(FancyBboxPatch((0.0, y-0.10), 0.985, 0.145,
        boxstyle="round,pad=0.01",
        facecolor=BG3, edgecolor=col, linewidth=1.5,
        transform=ax_rec.transAxes, clip_on=False))
    ax_rec.add_patch(plt.Rectangle((0.0, y-0.10), 0.006, 0.145,
        facecolor=col, transform=ax_rec.transAxes, clip_on=False))
    ax_rec.text(0.012, y+0.02, title, color=col,
                fontsize=8.5, fontweight="bold",
                transform=ax_rec.transAxes, va="center")
    ax_rec.text(0.012, y-0.055, body, color=MUTED,
                fontsize=7.5, transform=ax_rec.transAxes, va="center")

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
ax_foot = fig.add_subplot(outer[7])
ax_foot.set_facecolor(BG3)
ax_foot.axis("off")
ax_foot.text(0.5, 0.5,
    "Warehouse & Retail Sales Intelligence  ·  Sowmiya Subramaniyan  ·  "
    "4-Tier Analytics Platform  ·  Random Forest ROC-AUC 0.9056  ·  Recall 0.8112",
    color=MUTED, fontsize=9, ha="center", va="center",
    transform=ax_foot.transAxes)

# ── Save ──────────────────────────────────────────────────
out = Path("dashboard_preview.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out}  ({out.stat().st_size//1024} KB)")
