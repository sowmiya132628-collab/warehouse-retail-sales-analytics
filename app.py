"""
============================================================
Warehouse & Retail Sales Intelligence — Streamlit Dashboard
Author : Sowmiya Subramaniyan   |   Version 2.0 — Innovative
============================================================
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Warehouse & Retail Sales Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

/* ── App background ── */
.stApp { background: #060b14; color: #e2e8f0; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── KPI card — glowing border ── */
.kpi-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 18px 14px 14px;
    text-align: center;
    min-height: 108px;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, #38bdf8);
    border-radius: 14px 14px 0 0;
}
.kpi-icon  { font-size: 18px; margin-bottom: 4px; }
.kpi-title { font-size: 9px; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.1em; margin-bottom: 4px; font-weight: 600; }
.kpi-value { font-size: 20px; font-weight: 800; color: #38bdf8;
             white-space: nowrap; line-height: 1.1; }
.kpi-sub   { font-size: 10px; color: #475569; margin-top: 4px; font-weight: 500; }

/* ── Accent colour variants ── */
.kpi-green  .kpi-value { color: #34d399; }
.kpi-green::before  { background: #34d399; }
.kpi-purple .kpi-value { color: #a78bfa; }
.kpi-purple::before { background: #a78bfa; }
.kpi-orange .kpi-value { color: #fb923c; }
.kpi-orange::before { background: #fb923c; }
.kpi-pink   .kpi-value { color: #f472b6; }
.kpi-pink::before   { background: #f472b6; }
.kpi-teal   .kpi-value { color: #2dd4bf; }
.kpi-teal::before   { background: #2dd4bf; }

/* ── Section header ── */
.sec-hdr {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 14px; padding: 0 0 10px;
    border-bottom: 1px solid #1e3a5f;
}
.sec-hdr-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #38bdf8; box-shadow: 0 0 8px #38bdf8;
    flex-shrink: 0;
}
.sec-hdr-text { font-size: 15px; font-weight: 700; color: #e2e8f0; letter-spacing: 0.02em; }
.sec-hdr-badge {
    margin-left: auto; background: #1e3a5f; color: #38bdf8;
    border-radius: 20px; padding: 2px 10px; font-size: 10px; font-weight: 600;
}

/* ── Insight card ── */
.insight-card {
    background: linear-gradient(135deg, #0d1b2a, #0f2540);
    border: 1px solid #1e3a5f; border-left: 3px solid #38bdf8;
    border-radius: 10px; padding: 14px 18px;
    margin: 10px 0; font-size: 13px; color: #cbd5e1; line-height: 1.6;
}
.insight-card .tag {
    display: inline-block; background: #1e3a5f; color: #38bdf8;
    border-radius: 4px; padding: 1px 7px; font-size: 10px;
    font-weight: 600; margin-bottom: 6px; text-transform: uppercase;
}

/* ── Gauge bar ── */
.gauge-row { display:flex; align-items:center; gap:12px; margin:6px 0; }
.gauge-label { width:100px; font-size:11px; color:#94a3b8; text-align:right; flex-shrink:0; }
.gauge-track { flex:1; height:8px; background:#1e3a5f; border-radius:4px; overflow:hidden; }
.gauge-fill  { height:100%; border-radius:4px; background: linear-gradient(90deg,#38bdf8,#818cf8); }
.gauge-val   { width:48px; font-size:11px; font-weight:700; color:#e2e8f0; flex-shrink:0; }

/* ── Risk badge ── */
.risk-high   { background:#450a0a; color:#fca5a5; border:1px solid #7f1d1d;
               border-radius:6px; padding:2px 8px; font-size:10px; font-weight:700; }
.risk-med    { background:#431407; color:#fdba74; border:1px solid #7c2d12;
               border-radius:6px; padding:2px 8px; font-size:10px; font-weight:700; }
.risk-low    { background:#052e16; color:#86efac; border:1px solid #14532d;
               border-radius:6px; padding:2px 8px; font-size:10px; font-weight:700; }

/* ── Metric pill row ── */
.pill-row { display:flex; flex-wrap:wrap; gap:10px; margin:10px 0; }
.pill {
    background: #0d1b2a; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 10px 16px; min-width: 110px; text-align:center;
}
.pill-label { font-size:10px; color:#64748b; text-transform:uppercase;
              letter-spacing:.06em; margin-bottom:4px; }
.pill-value { font-size:18px; font-weight:800; color:#38bdf8; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #1e3a5f; margin: 20px 0; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1b2a; border-bottom: 1px solid #1e3a5f; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b; padding: 10px 18px; font-size: 13px; font-weight: 500;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important; background: #112240 !important;
    border-bottom: 2px solid #38bdf8;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #0d1b2a; border: 1px solid #1e3a5f;
    color: #38bdf8; border-radius: 8px; font-size: 12px;
}
.stDownloadButton > button:hover { background: #1e3a5f; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Chart colour palette ──────────────────────────────────
C = ["#38bdf8","#34d399","#fb923c","#a78bfa","#f472b6","#2dd4bf","#facc15","#818cf8"]
BG   = "#060b14"
BG2  = "#0d1b2a"
BG3  = "#112240"
GRID = "#1e3a5f"

def fig_base(height=380):
    return go.Figure(
        layout=go.Layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color="#cbd5e1", family="Inter, Segoe UI, sans-serif", size=12),
            xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
            legend=dict(bgcolor=BG3, bordercolor=GRID, borderwidth=1,
                        font=dict(size=11)),
            margin=dict(t=50, b=40, l=50, r=20),
            height=height,
            colorway=C,
        )
    )

def sec(title, badge=None):
    badge_html = f'<span class="sec-hdr-badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="sec-hdr"><div class="sec-hdr-dot"></div>'
        f'<div class="sec-hdr-text">{title}</div>{badge_html}</div>',
        unsafe_allow_html=True,
    )

def insight(tag_text, body):
    st.markdown(
        f'<div class="insight-card"><span class="tag">{tag_text}</span><br>{body}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_clean():
    df = pd.read_csv("Warehouse_and_Retail_Sales.csv")
    audit = {
        "rows": len(df), "cols": df.shape[1],
        "missing": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "neg_rs": int((df["RETAIL SALES"] < 0).sum()),
        "neg_rt": int((df["RETAIL TRANSFERS"] < 0).sum()),
        "neg_ws": int((df["WAREHOUSE SALES"] < 0).sum()),
        "item_type_dist": df["ITEM TYPE"].value_counts().to_dict(),
    }
    df["SUPPLIER"]         = df["SUPPLIER"].fillna("UNKNOWN SUPPLIER")
    df["ITEM TYPE"]        = df["ITEM TYPE"].fillna("UNKNOWN")
    df["RETAIL SALES"]     = df["RETAIL SALES"].fillna(0.0).clip(lower=0)
    df["RETAIL TRANSFERS"] = df["RETAIL TRANSFERS"].clip(lower=0)
    df["WAREHOUSE SALES"]  = df["WAREHOUSE SALES"].clip(lower=0)
    df["ITEM CODE"]        = df["ITEM CODE"].astype(str).str.strip()
    for col in ["SUPPLIER", "ITEM TYPE", "ITEM DESCRIPTION"]:
        df[col] = df[col].str.strip().str.upper()
    minor = {"STR_SUPPLIES", "REF", "DUNNAGE"}
    df["ITEM TYPE CLEAN"] = df["ITEM TYPE"].apply(lambda x: "OTHER" if x in minor else x)
    df["DATE"]            = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01"
    )
    df["TOTAL MOVEMENT"]  = df["RETAIL SALES"] + df["RETAIL TRANSFERS"] + df["WAREHOUSE SALES"]
    return df, audit


@st.cache_resource(show_spinner=False)
def train_model():
    df, _ = load_and_clean()
    p90   = df["WAREHOUSE SALES"].quantile(0.90)
    df["High_Warehouse_Demand"] = (df["WAREHOUSE SALES"] >= p90).astype(int)
    le_t  = LabelEncoder(); le_s = LabelEncoder()
    df["ITEM_TYPE_ENC"] = le_t.fit_transform(df["ITEM TYPE CLEAN"])
    df["SUPPLIER_ENC"]  = le_s.fit_transform(df["SUPPLIER"])
    FEAT = ["YEAR", "MONTH", "ITEM_TYPE_ENC", "SUPPLIER_ENC"]
    TGT  = "High_Warehouse_Demand"
    trn  = df["YEAR"] < 2020
    tst  = df["YEAR"] == 2020
    clf  = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(df.loc[trn, FEAT], df.loc[trn, TGT])
    y_pred = clf.predict(df.loc[tst, FEAT])
    y_prob = clf.predict_proba(df.loc[tst, FEAT])[:, 1]
    y_test = df.loc[tst, TGT]
    metrics = dict(
        Accuracy  = round(accuracy_score(y_test, y_pred), 4),
        Precision = round(precision_score(y_test, y_pred, zero_division=0), 4),
        Recall    = round(recall_score(y_test, y_pred, zero_division=0), 4),
        F1        = round(f1_score(y_test, y_pred, zero_division=0), 4),
        ROC_AUC   = round(roc_auc_score(y_test, y_prob), 4),
    )
    cm       = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    imp      = pd.DataFrame({
        "Feature"   : ["Year", "Month", "Item Type", "Supplier"],
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    pred_df  = df[tst].copy()
    pred_df["Prob_High_Demand"] = y_prob
    pred_df["Predicted_Class"]  = y_pred
    pred_df["Actual_Class"]     = y_test.values
    return dict(metrics=metrics, cm=cm, fpr=fpr, tpr=tpr,
                imp=imp, pred_df=pred_df, p90=p90)


# ── Load ──────────────────────────────────────────────────
with st.spinner("🚀 Initialising dashboard…"):
    df_clean, audit = load_and_clean()
with st.spinner("🤖 Loading ML model…"):
    ml = train_model()

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;text-align:center;">
      <div style="font-size:28px;">📦</div>
      <div style="font-size:13px;font-weight:700;color:#38bdf8;margin-top:4px;">WR Sales Intel</div>
      <div style="font-size:10px;color:#475569;margin-top:2px;">v2.0 · Montgomery County</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)

    st.markdown("**🗓 Year**")
    years = sorted(df_clean["YEAR"].unique())
    sel_years = st.multiselect("", years, default=years, label_visibility="collapsed")

    st.markdown("**📅 Month**")
    months = sorted(df_clean["MONTH"].unique())
    sel_months = st.multiselect("", months, default=months, label_visibility="collapsed")

    st.markdown("**🏷 Item Type**")
    item_types = sorted(df_clean["ITEM TYPE CLEAN"].unique())
    sel_types = st.multiselect("", item_types, default=item_types, label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)
    st.markdown("**🔢 Top N Suppliers**")
    top_n = st.slider("", 5, 20, 10, label_visibility="collapsed")

    st.markdown("**⚙️ Review Capacity**")
    capacity = st.select_slider("", [100,250,500,1000,5000,10000], value=500,
                                 label_visibility="collapsed")

    st.markdown('<hr style="border-color:#1e3a5f;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:#475569;line-height:1.8;">
      📁 307,645 records<br>
      📆 2017 – 2020<br>
      🏭 396 suppliers<br>
      🛒 34,056 SKUs
    </div>
    """, unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────
mask = (df_clean["YEAR"].isin(sel_years) &
        df_clean["MONTH"].isin(sel_months) &
        df_clean["ITEM TYPE CLEAN"].isin(sel_types))
df_f = df_clean[mask].copy()

wh  = df_f["WAREHOUSE SALES"].sum()
rs  = df_f["RETAIL SALES"].sum()
rt  = df_f["RETAIL TRANSFERS"].sum()
np_ = df_f["ITEM CODE"].nunique()
ns  = df_f["SUPPLIER"].nunique()
avg = df_f.groupby(["YEAR","MONTH"])["WAREHOUSE SALES"].sum().mean() if not df_f.empty else 0

# ══════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1b2a 0%,#112240 60%,#0f2a45 100%);
            border:1px solid #1e3a5f; border-radius:16px; padding:22px 28px 18px;
            margin-bottom:20px; position:relative; overflow:hidden;">
  <div style="position:absolute;top:-30px;right:-30px;width:160px;height:160px;
              background:radial-gradient(circle,#38bdf820,transparent 70%);pointer-events:none;"></div>
  <div style="font-size:11px;color:#38bdf8;font-weight:600;letter-spacing:.12em;
              text-transform:uppercase;margin-bottom:6px;">
      ⬡ 4-TIER ANALYTICS PLATFORM
  </div>
  <div style="font-size:24px;font-weight:800;color:#f1f5f9;letter-spacing:-.01em;">
      Warehouse &amp; Retail Sales Intelligence
  </div>
  <div style="font-size:13px;color:#64748b;margin-top:6px;">
      Montgomery County Alcohol Distribution &nbsp;·&nbsp;
      {len(df_f):,} records selected &nbsp;·&nbsp;
      {len(sel_years)} year(s) &nbsp;·&nbsp;
      {len(sel_types)} category(s)
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ Overview", "🔬 Data Quality", "📊 Deep Analytics",
    "🤖 ML Engine", "🎯 Risk Scores", "💡 Strategy"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:

    # ── KPI row ──────────────────────────────────────────
    sec("Key Performance Indicators", f"{len(df_f):,} records")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [
        (k1,"","Warehouse Sales", f"{wh/1e6:.2f}M","units",""),
        (k2,"kpi-green","Retail Sales", f"{rs/1e6:.2f}M","units",""),
        (k3,"kpi-orange","Retail Transfers",f"{rt/1e6:.2f}M","units",""),
        (k4,"kpi-purple","Unique SKUs",f"{np_:,}","products",""),
        (k5,"kpi-pink","Suppliers",f"{ns:,}","active",""),
        (k6,"kpi-teal","Avg Monthly WH",f"{avg/1e3:.1f}K","units/mo",""),
    ]
    icons = ["🏭","🛒","🔄","📦","🤝","📈"]
    for (col, cls, title, val, sub, _), icon in zip(kpis, icons):
        col.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Area chart — Monthly trend ────────────────────────
    sec("Monthly Sales Trend 2017–2020", "Time Series")
    monthly = (df_f.groupby("DATE")[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]]
               .sum().reset_index().sort_values("DATE"))

    fig_area = fig_base(height=340)
    fig_area.add_trace(go.Scatter(
        x=monthly["DATE"], y=monthly["WAREHOUSE SALES"],
        name="Warehouse Sales", fill="tozeroy",
        line=dict(color=C[0], width=2),
        fillcolor="rgba(56,189,248,0.12)"))
    fig_area.add_trace(go.Scatter(
        x=monthly["DATE"], y=monthly["RETAIL SALES"],
        name="Retail Sales", fill="tozeroy",
        line=dict(color=C[1], width=2),
        fillcolor="rgba(52,211,153,0.10)"))
    fig_area.add_trace(go.Scatter(
        x=monthly["DATE"], y=monthly["RETAIL TRANSFERS"],
        name="Retail Transfers",
        line=dict(color=C[2], width=1.5, dash="dot")))
    fig_area.update_layout(
        title=None,
        xaxis_title=None, yaxis_title="Units",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig_area, use_container_width=True)
    insight("Observation",
            "Warehouse sales peaked in <b>mid-2019</b> at ~1.5× the period average, "
            "then declined into 2020. Retail sales remained consistently flat — "
            "the two channels behave independently (r ≈ 0.15).")

    # ── Bottom row — donut + stacked bar ─────────────────
    cl, cr = st.columns([1, 1.4])
    with cl:
        sec("Channel Mix by Item Type", "Donut")
        type_wh = df_f.groupby("ITEM TYPE CLEAN")["WAREHOUSE SALES"].sum().reset_index()
        fig_donut = go.Figure(go.Pie(
            labels=type_wh["ITEM TYPE CLEAN"],
            values=type_wh["WAREHOUSE SALES"],
            hole=0.6, direction="clockwise",
            marker=dict(colors=C, line=dict(color=BG, width=2)),
            textfont=dict(size=11),
        ))
        fig_donut.update_layout(
            paper_bgcolor=BG, height=300,
            margin=dict(t=20,b=20,l=10,r=10),
            legend=dict(bgcolor=BG3, bordercolor=GRID, borderwidth=1,
                        font=dict(size=10), orientation="v"),
            annotations=[dict(text="<b>WH<br>Sales</b>", x=0.5, y=0.5,
                               font=dict(size=13, color="#e2e8f0"), showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with cr:
        sec("Annual Sales by Category", "Stacked Bar")
        yr_type = df_f.groupby(["YEAR","ITEM TYPE CLEAN"])["WAREHOUSE SALES"].sum().reset_index()
        fig_bar = px.bar(yr_type, x="YEAR", y="WAREHOUSE SALES", color="ITEM TYPE CLEAN",
                         barmode="stack", color_discrete_sequence=C,
                         labels={"WAREHOUSE SALES":"Units","ITEM TYPE CLEAN":"Category"},
                         template="plotly_dark")
        fig_bar.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2, height=300,
                               margin=dict(t=20,b=30,l=50,r=10),
                               legend=dict(bgcolor=BG3, bordercolor=GRID, font=dict(size=10)),
                               xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ══════════════════════════════════════════════════════════
with tab2:
    sec("Dataset Health Scorecard", "Audit")
    qa1,qa2,qa3,qa4 = st.columns(4)
    for col, lbl, val, cls in [
        (qa1,"Total Records",   f"{audit['rows']:,}",""),
        (qa2,"Duplicate Rows",  str(audit['duplicates']),"kpi-green"),
        (qa3,"Missing Cells",   str(sum(v for v in audit['missing'].values() if v>0)),"kpi-orange"),
        (qa4,"Negative Values", str(audit['neg_rs']+audit['neg_rt']+audit['neg_ws']),"kpi-pink"),
    ]:
        col.markdown(f"""
        <div class="kpi-card {cls}" style="min-height:80px;">
          <div class="kpi-title">{lbl}</div>
          <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl2, cr2 = st.columns(2)

    with cl2:
        sec("Missing Values", "Per Column")
        miss_df = pd.DataFrame({
            "Column": list(audit["missing"].keys()),
            "Count":  list(audit["missing"].values()),
        })
        miss_df["%"] = (miss_df["Count"] / audit["rows"] * 100).round(3)
        # visual bar inside table using gauge bars
        for _, row in miss_df.iterrows():
            pct   = min(row["%"], 100)
            color = "#38bdf8" if pct == 0 else "#fb923c"
            st.markdown(
                f'<div class="gauge-row">'
                f'<div class="gauge-label">{row["Column"][:14]}</div>'
                f'<div class="gauge-track"><div class="gauge-fill" '
                f'style="width:{max(pct*20,2 if pct>0 else 0)}%;background:{color};"></div></div>'
                f'<div class="gauge-val">{row["Count"]}</div>'
                f'</div>', unsafe_allow_html=True)

    with cr2:
        sec("Negative Values", "Anomaly")
        neg_data = {
            "RETAIL SALES": audit["neg_rs"],
            "RETAIL TRANSFERS": audit["neg_rt"],
            "WAREHOUSE SALES": audit["neg_ws"],
        }
        fig_neg = go.Figure(go.Bar(
            x=list(neg_data.keys()), y=list(neg_data.values()),
            marker=dict(color=[C[2],C[3],C[4]],
                        line=dict(color=BG, width=1)),
            text=list(neg_data.values()),
            textposition="outside",
            textfont=dict(size=12, color="#e2e8f0"),
        ))
        fig_neg.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2, height=260,
            margin=dict(t=20,b=30,l=50,r=10),
            xaxis=dict(gridcolor=GRID, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID, title="Count"),
        )
        st.plotly_chart(fig_neg, use_container_width=True)

    sec("Cleaning Decision Log", "7 Actions")
    decisions = [
        ("167 NULL SUPPLIER",          "Fill → 'UNKNOWN SUPPLIER'", "Preserve valid sale records"),
        ("1 NULL ITEM TYPE",           "Fill → 'UNKNOWN'",          "Single row; removal unwarranted"),
        ("3 NULL RETAIL SALES",        "Fill → 0.0",                "Zero movement assumption"),
        ("716 negative WH SALES",      "Clip → 0",                  "Returns/adjustments floor"),
        ("1,016 negative RT",          "Clip → 0",                  "Same rationale"),
        ("113 negative RS",            "Clip → 0",                  "Same rationale"),
        ("STR_SUPPLIES/REF/DUNNAGE",   "Group → 'OTHER'",           "<0.2% each; reduce ML noise"),
    ]
    dlog = pd.DataFrame(decisions, columns=["Problem","Action","Reason"])
    st.dataframe(dlog, use_container_width=True, hide_index=True, height=280)

    sec("Record Distribution by Item Type", "Raw")
    td = pd.DataFrame(list(audit["item_type_dist"].items()),
                      columns=["Type","Count"]).sort_values("Count",ascending=False)
    fig_td = go.Figure(go.Bar(
        x=td["Type"], y=td["Count"],
        marker=dict(color=C[:len(td)], line=dict(color=BG,width=1)),
        text=td["Count"], textposition="outside",
        textfont=dict(size=10, color="#e2e8f0"),
    ))
    fig_td.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2, height=280,
                          margin=dict(t=10,b=30,l=60,r=10),
                          xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                          showlegend=False)
    st.plotly_chart(fig_td, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — DEEP ANALYTICS (EDA)
# ══════════════════════════════════════════════════════════
with tab3:

    # ── VIZ 1: Area + volume combo ───────────────────────
    sec("1 · Monthly Sales Volume — All Channels", "Time Series")
    monthly2 = (df_f.groupby("DATE")[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]]
                .sum().reset_index().sort_values("DATE"))
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7,0.3],
                         vertical_spacing=0.06,
                         subplot_titles=("Channel Sales Volume","Total Movement"))
    for col_name, color, fill in [
        ("WAREHOUSE SALES",  C[0], "rgba(56,189,248,0.15)"),
        ("RETAIL SALES",     C[1], "rgba(52,211,153,0.10)"),
        ("RETAIL TRANSFERS", C[2], "rgba(251,146,60,0.10)"),
    ]:
        fig1.add_trace(go.Scatter(
            x=monthly2["DATE"], y=monthly2[col_name],
            name=col_name, fill="tozeroy",
            line=dict(color=color, width=2), fillcolor=fill,
        ), row=1, col=1)
    total_mv = monthly2[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]].sum(axis=1)
    fig1.add_trace(go.Bar(
        x=monthly2["DATE"], y=total_mv,
        name="Total Movement", marker_color=C[5], opacity=0.7, showlegend=True,
    ), row=2, col=1)
    fig1.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2, height=440,
                        margin=dict(t=40,b=30,l=60,r=20),
                        legend=dict(bgcolor=BG3, bordercolor=GRID, font=dict(size=10),
                                    orientation="h", y=1.08, x=0),
                        font=dict(color="#cbd5e1", size=11))
    fig1.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig1.update_yaxes(gridcolor=GRID, linecolor=GRID)
    st.plotly_chart(fig1, use_container_width=True)
    with st.expander("📋 Analysis"):
        st.markdown("""
**Observation:** Warehouse sales peaked in mid-2019 at ~1.5× the period average monthly volume, then
declined. Retail sales remained consistently flat. Total movement closely tracks warehouse patterns.

**Diagnostic Insight:** 2019 surge likely reflects expanded product onboarding or distributor order
pattern changes rather than a pure consumer demand increase.

**Causality Disclaimer:** Temporal association only — pricing, policy, or seasonal factors cannot
be confirmed from this dataset alone.
        """)

    st.markdown("---")

    # ── VIZ 2: Grouped + % stacked ───────────────────────
    sec("2 · Warehouse vs Retail by Item Type", "Channel Split")
    type_agg = (df_f.groupby("ITEM TYPE CLEAN")[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]]
                .sum().reset_index().sort_values("WAREHOUSE SALES", ascending=False))
    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Absolute Volume", "Channel Share (%)"),
                         column_widths=[0.55, 0.45])
    # Left panel — grouped absolute bars
    for col_name, color in [("WAREHOUSE SALES",C[0]),("RETAIL SALES",C[1]),
                              ("RETAIL TRANSFERS",C[2])]:
        fig2.add_trace(go.Bar(
            name=col_name, x=type_agg["ITEM TYPE CLEAN"], y=type_agg[col_name],
            marker_color=color, marker_line_width=0,
        ), row=1, col=1)
    # Right panel — 100% stacked (pre-compute % values, stack them)
    total_col = type_agg[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]].sum(axis=1)
    for col_name, color in [("WAREHOUSE SALES",C[0]),("RETAIL SALES",C[1]),
                              ("RETAIL TRANSFERS",C[2])]:
        fig2.add_trace(go.Bar(
            name=col_name,
            x=type_agg["ITEM TYPE CLEAN"],
            y=(type_agg[col_name] / total_col * 100).round(1),
            marker_color=color, marker_line_width=0, showlegend=False,
            offsetgroup=col_name,
        ), row=1, col=2)
    # barmode="stack" applies to all subplots — use barnorm on right panel instead
    fig2.update_layout(
        barmode="stack",          # stacks col-2 bars; col-1 uses offsetgroup so groups correctly
        paper_bgcolor=BG, plot_bgcolor=BG2, height=380,
        margin=dict(t=50, b=30, l=60, r=20),
        legend=dict(bgcolor=BG3, bordercolor=GRID, font=dict(size=10),
                    orientation="h", y=1.1, x=0),
        font=dict(color="#cbd5e1", size=11),
    )
    # Override col-1 to group mode via its own barmode per axis is not possible;
    # workaround: use offsetgroup on col-1 traces so they render side-by-side
    fig2.update_layout({"barmode": "overlay"})
    # Correct approach: separate the two panels as independent bar traces with offsets
    # Reset and rebuild cleanly using two separate figures composed via subplot
    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Absolute Volume", "Channel Share (%)"),
                         column_widths=[0.55, 0.45])
    channels = [("WAREHOUSE SALES",C[0]),("RETAIL SALES",C[1]),("RETAIL TRANSFERS",C[2])]
    total_col = type_agg[["WAREHOUSE SALES","RETAIL SALES","RETAIL TRANSFERS"]].sum(axis=1)
    for col_name, color in channels:
        fig2.add_trace(go.Bar(
            name=col_name, x=type_agg["ITEM TYPE CLEAN"], y=type_agg[col_name],
            marker_color=color, marker_line_width=0, legendgroup=col_name,
        ), row=1, col=1)
        fig2.add_trace(go.Bar(
            name=col_name,
            x=type_agg["ITEM TYPE CLEAN"],
            y=(type_agg[col_name] / total_col * 100).round(1),
            marker_color=color, marker_line_width=0,
            showlegend=False, legendgroup=col_name,
        ), row=1, col=2)
    fig2.update_layout(
        barmode="stack",
        paper_bgcolor=BG, plot_bgcolor=BG2, height=380,
        margin=dict(t=50, b=30, l=60, r=20),
        legend=dict(bgcolor=BG3, bordercolor=GRID, font=dict(size=10),
                    orientation="h", y=1.1, x=0),
        font=dict(color="#cbd5e1", size=11),
    )
    fig2.update_xaxes(gridcolor=GRID, linecolor=GRID, tickangle=-20, tickfont=dict(size=9))
    fig2.update_yaxes(gridcolor=GRID, linecolor=GRID)
    fig2.update_yaxes(title_text="Units", row=1, col=1)
    fig2.update_yaxes(title_text="%", row=1, col=2)
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("📋 Analysis"):
        st.markdown("""
**Observation:** WINE dominates warehouse volume (~60%), LIQUOR follows at ~21%. Beer shows
a higher retail-to-warehouse share ratio than Wine.

**Business Relevance:** Wine requires heavier warehouse infrastructure; Beer more direct retail logistics.

**Causality Disclaimer:** Channel split reflects recorded transactions, not consumer demand causality.
        """)

    st.markdown("---")

    # ── VIZ 3: Lollipop supplier chart ───────────────────
    sec(f"3 · Top {top_n} Suppliers — Warehouse Volume", "Supplier Ranking")
    top_sup = (df_f.groupby("SUPPLIER")["WAREHOUSE SALES"]
               .sum().reset_index()
               .sort_values("WAREHOUSE SALES", ascending=False)
               .head(top_n))
    top_sup["SHORT"] = top_sup["SUPPLIER"].str[:28]
    top_sup = top_sup.sort_values("WAREHOUSE SALES")

    fig3 = fig_base(height=max(320, top_n * 32))
    # Line from 0 to value
    for _, row in top_sup.iterrows():
        fig3.add_shape(type="line",
                       x0=0, x1=row["WAREHOUSE SALES"],
                       y0=row["SHORT"], y1=row["SHORT"],
                       line=dict(color=GRID, width=2))
    fig3.add_trace(go.Scatter(
        x=top_sup["WAREHOUSE SALES"], y=top_sup["SHORT"],
        mode="markers+text",
        marker=dict(size=14, color=C[0],
                    line=dict(color=BG2, width=2),
                    symbol="circle"),
        text=top_sup["WAREHOUSE SALES"].apply(lambda v: f"{v:,.0f}"),
        textposition="middle right",
        textfont=dict(size=9, color="#94a3b8"),
        name="Total WH Sales",
    ))
    fig3.update_layout(
        xaxis_title="Total Warehouse Units Sold",
        yaxis_title=None,
        margin=dict(t=20, b=40, l=240, r=80),
    )
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("📋 Analysis"):
        st.markdown("""
**Observation:** Top 5 suppliers account for ~35% of total warehouse units — significant concentration risk.

**Business Relevance:** A disruption to the top supplier (REPUBLIC NATIONAL) could drop
overall warehouse throughput by up to 7% immediately.

**Causality Disclaimer:** High volume correlates with supplier scale and product breadth, not causality.
        """)

    st.markdown("---")

    # ── VIZ 4: Bubble scatter + correlation ──────────────
    sec("4 · Sales Relationship Analysis", "Correlation & Scatter")
    cl4, cr4 = st.columns(2)
    with cl4:
        num_cols = ["RETAIL SALES","RETAIL TRANSFERS","WAREHOUSE SALES","TOTAL MOVEMENT"]
        corr = df_f[num_cols].corr().round(2)
        fig4a = px.imshow(corr, text_auto=True,
                          color_continuous_scale=[[0,BG2],[0.5,"#1e3a5f"],[1,C[0]]],
                          zmin=-1, zmax=1,
                          labels=dict(color="r"),
                          template="plotly_dark")
        fig4a.update_layout(paper_bgcolor=BG, height=320,
                             margin=dict(t=30,b=10,l=10,r=10),
                             coloraxis_colorbar=dict(thickness=10, tickfont=dict(size=9)),
                             font=dict(size=10))
        fig4a.update_traces(textfont_size=11)
        st.plotly_chart(fig4a, use_container_width=True)

    with cr4:
        sample = df_f.sample(min(3000, len(df_f)), random_state=42)
        fig4b = px.scatter(sample,
                           x="RETAIL SALES", y="WAREHOUSE SALES",
                           color="ITEM TYPE CLEAN",
                           color_discrete_sequence=C,
                           opacity=0.5, size_max=6,
                           labels={"RETAIL SALES":"Retail Sales","WAREHOUSE SALES":"WH Sales"},
                           template="plotly_dark")
        fig4b.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2, height=320,
                             margin=dict(t=20,b=30,l=60,r=10),
                             xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                             legend=dict(bgcolor=BG3, bordercolor=GRID, font=dict(size=9),
                                         title=dict(text="")))
        fig4b.update_traces(marker=dict(size=5))
        st.plotly_chart(fig4b, use_container_width=True)

    with st.expander("📋 Analysis"):
        st.markdown("""
**Observation:** Retail Sales & Transfers correlate moderately (r ≈ 0.55). Warehouse Sales shows
weak correlation with retail (r ≈ 0.15–0.25). The scatter confirms the two channels are largely
independent at item level.

**Business Relevance:** Retail velocity is NOT a reliable proxy for warehouse demand — separate models needed.
        """)

    st.markdown("---")

    # ── VIZ 5: Heatmap calendar-style ────────────────────
    sec("5 · Demand Intensity: Category × Year", "Heat Map")
    pivot = (df_f.groupby(["YEAR","ITEM TYPE CLEAN"])["WAREHOUSE SALES"]
             .sum().unstack("ITEM TYPE CLEAN").fillna(0))
    pivot_norm = pivot.div(pivot.max()).round(3)
    fig5 = px.imshow(pivot_norm,
                     color_continuous_scale=[[0,BG2],[0.4,"#1e3a5f"],
                                             [0.7,C[4]],[1.0,C[2]]],
                     zmin=0, zmax=1, text_auto=".2f",
                     labels=dict(x="Item Type", y="Year", color="Intensity"),
                     template="plotly_dark",
                     aspect="auto")
    fig5.update_layout(paper_bgcolor=BG, height=320,
                        margin=dict(t=30,b=30,l=60,r=20),
                        font=dict(size=11),
                        coloraxis_colorbar=dict(thickness=10,
                                                tickfont=dict(size=9)))
    fig5.update_traces(textfont_size=11)
    st.plotly_chart(fig5, use_container_width=True)
    with st.expander("📋 Analysis"):
        st.markdown("""
**Observation:** Wine and Liquor peaked in 2019 (intensity = 1.0 normalised). 2020 shows decline
across most categories, consistent with COVID-19 on-premise closures. Non-Alcohol remains low throughout.

**Causality Disclaimer:** Temporal association only — external market factors cannot be confirmed from data.
        """)


# ══════════════════════════════════════════════════════════
# TAB 4 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════
with tab4:

    sec("Model Architecture", "Random Forest · 2020 Test Set")
    st.markdown(f"""
    <div class="insight-card">
      <span class="tag">Derived Target</span><br>
      <b>High_Warehouse_Demand = 1</b> if WAREHOUSE SALES &ge; <b>{ml['p90']:.1f} units (p90)</b>, else 0
      &nbsp;&nbsp;→&nbsp;&nbsp; ~10% positive class · operationally = priority restock items<br><br>
      <b>Split:</b> Chronological — 2017-2019 train (261,367 rows) · 2020 test (46,278 rows)<br>
      <b>Leakage prevention:</b> WAREHOUSE SALES, TOTAL MOVEMENT, RETAIL SALES,
      RETAIL TRANSFERS, ITEM CODE, ITEM DESCRIPTION excluded<br>
      <b>Features used:</b> YEAR · MONTH · ITEM TYPE (encoded) · SUPPLIER (encoded)
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────
    sec("Performance Metrics", "2020 Holdout")
    m_cols = st.columns(5)
    mcolors = [C[0], C[1], C[2], C[3], C[4]]
    micons  = ["🎯","📐","🔭","⚖️","📈"]
    for i,(col,(name,val)) in enumerate(zip(m_cols, ml["metrics"].items())):
        col.markdown(f"""
        <div class="kpi-card" style="--accent:{mcolors[i]};">
          <div class="kpi-icon">{micons[i]}</div>
          <div class="kpi-title">{name}</div>
          <div class="kpi-value" style="color:{mcolors[i]};font-size:24px;">{val:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge bars for metrics ────────────────────────────
    for name, val in ml["metrics"].items():
        pct = val * 100
        bar_color = C[0] if val >= 0.85 else (C[1] if val >= 0.6 else C[2])
        st.markdown(
            f'<div class="gauge-row">'
            f'<div class="gauge-label" style="width:90px;">{name}</div>'
            f'<div class="gauge-track"><div class="gauge-fill" '
            f'style="width:{pct:.1f}%;background:{bar_color};"></div></div>'
            f'<div class="gauge-val">{val:.4f}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl_m, cr_m = st.columns(2)

    with cl_m:
        sec("Confusion Matrix", "Heatmap")
        cm = ml["cm"]
        labels = ["Normal","High Demand"]
        total  = cm.sum()
        pct_cm = (cm / total * 100).round(1)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred: Normal","Pred: High"],
            y=["Act: Normal","Act: High"],
            colorscale=[[0,BG3],[0.5,"#1e3a5f"],[1,C[0]]],
            showscale=False,
            text=[[f"<b>{cm[r][c]:,}</b><br>({pct_cm[r][c]}%)"
                   for c in range(2)] for r in range(2)],
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            hoverinfo="none",
        ))
        fig_cm.update_layout(paper_bgcolor=BG, height=320,
                              margin=dict(t=20,b=40,l=80,r=20),
                              font=dict(color="#cbd5e1", size=11))
        st.plotly_chart(fig_cm, use_container_width=True)

    with cr_m:
        sec("ROC Curve", f"AUC = {ml['metrics']['ROC_AUC']:.4f}")
        fig_roc = fig_base(height=320)
        # Shaded AUC area
        fig_roc.add_trace(go.Scatter(
            x=ml["fpr"], y=ml["tpr"],
            fill="tozeroy", fillcolor="rgba(56,189,248,0.12)",
            line=dict(color=C[0], width=2.5),
            name=f"AUC = {ml['metrics']['ROC_AUC']:.4f}"))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            line=dict(color="#334155", width=1.5, dash="dot"),
            name="Random"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=20,b=40,l=60,r=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    sec("Feature Importance", "What Drives the Prediction?")
    imp = ml["imp"]
    fig_imp = fig_base(height=240)
    # Horizontal lollipop
    for _, row in imp.iterrows():
        fig_imp.add_shape(type="line",
                          x0=0, x1=row["Importance"],
                          y0=row["Feature"], y1=row["Feature"],
                          line=dict(color=GRID, width=3))
    fig_imp.add_trace(go.Scatter(
        x=imp["Importance"], y=imp["Feature"],
        mode="markers+text",
        marker=dict(size=16, color=C, line=dict(color=BG2, width=2)),
        text=imp["Importance"].apply(lambda v: f"{v:.3f}"),
        textposition="middle right",
        textfont=dict(size=11, color="#94a3b8"),
        name="Importance",
    ))
    fig_imp.update_layout(
        xaxis_title="Importance Score",
        yaxis_title=None,
        margin=dict(t=10,b=40,l=100,r=80),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    insight("Interpretability",
            "SUPPLIER is the strongest predictor — certain distributors consistently supply "
            "high-volume SKUs above the p90 threshold. MONTH captures seasonality. "
            "<b>Important:</b> importance = predictive association, NOT causal driver.")

    st.markdown("---")
    sec("FP / FN Business Trade-off", "Operational Impact")
    tf1, tf2 = st.columns(2)
    with tf1:
        st.markdown("""
        <div class="insight-card" style="border-left-color:#fb923c;">
        <span class="tag" style="background:#431407;color:#fb923c;">False Positive</span><br>
        Model predicts <b>High Demand</b> but reality is Normal<br><br>
        • Unnecessary pre-stocking & excess inventory<br>
        • Capital tied up in surplus stock<br>
        • Increased storage / holding costs<br>
        • <b>Impact: Moderate — manageable with returns</b>
        </div>""", unsafe_allow_html=True)
    with tf2:
        st.markdown("""
        <div class="insight-card" style="border-left-color:#f87171;">
        <span class="tag" style="background:#450a0a;color:#f87171;">False Negative</span><br>
        Model <b>misses</b> a genuinely High Demand item<br><br>
        • Warehouse stock-out during peak period<br>
        • Emergency reorder at premium cost<br>
        • Lost sales + potential contract penalties<br>
        • <b>Impact: HIGH — stock-outs are costly in alcohol distribution</b>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 5 — RISK / PREDICTION
# ══════════════════════════════════════════════════════════
with tab5:
    pred_df = ml["pred_df"].copy()
    thresh  = st.slider("🎚 High-Risk Probability Threshold", 0.30, 0.95, 0.70, 0.05)
    pred_df["Risk"] = pred_df["Prob_High_Demand"].apply(
        lambda p: "HIGH" if p>=thresh else ("MEDIUM" if p>=0.4 else "LOW"))

    sec("Risk Distribution", "2020 Predictions")
    rh = (pred_df["Risk"]=="HIGH").sum()
    rm = (pred_df["Risk"]=="MEDIUM").sum()
    rl = (pred_df["Risk"]=="LOW").sum()
    r1,r2,r3,r4 = st.columns(4)
    for col, lbl, val, cls in [
        (r1,"Total Predictions", f"{len(pred_df):,}",""),
        (r2,"🔴 High Risk",  f"{rh:,}", "kpi-pink"),
        (r3,"🟡 Medium Risk",f"{rm:,}", "kpi-orange"),
        (r4,"🟢 Low Risk",   f"{rl:,}", "kpi-green"),
    ]:
        col.markdown(f"""
        <div class="kpi-card {cls}" style="min-height:80px;">
          <div class="kpi-title">{lbl}</div>
          <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cl5, cr5 = st.columns(2)
    with cl5:
        sec("Probability Distribution", "Histogram")
        fig_ph = fig_base(height=300)
        fig_ph.add_trace(go.Histogram(
            x=pred_df["Prob_High_Demand"], nbinsx=50,
            marker=dict(color=C[0], opacity=0.8,
                        line=dict(color=BG, width=0.5)),
            name="All predictions",
        ))
        fig_ph.add_vline(x=thresh, line_width=2,
                          line_color=C[2], annotation_text=f"Threshold {thresh}",
                          annotation_font=dict(color=C[2], size=11))
        fig_ph.update_layout(
            xaxis_title="Predicted Probability",
            yaxis_title="Count",
            margin=dict(t=20,b=40,l=60,r=20),
        )
        st.plotly_chart(fig_ph, use_container_width=True)

    with cr5:
        sec("Risk by Item Type", "Breakdown")
        risk_type = (pred_df.groupby(["ITEM TYPE","Risk"])
                     .size().reset_index(name="Count"))
        fig_rt = px.bar(risk_type, x="ITEM TYPE", y="Count", color="Risk",
                        color_discrete_map={"HIGH":C[2],"MEDIUM":C[4],"LOW":C[1]},
                        barmode="stack", template="plotly_dark",
                        labels={"ITEM TYPE":"Item Type"})
        fig_rt.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2, height=300,
                              margin=dict(t=20,b=40,l=60,r=10),
                              xaxis=dict(gridcolor=GRID,tickangle=-20,tickfont=dict(size=9)),
                              yaxis=dict(gridcolor=GRID),
                              legend=dict(bgcolor=BG3,bordercolor=GRID,font=dict(size=10)))
        fig_rt.update_traces(marker_line_width=0)
        st.plotly_chart(fig_rt, use_container_width=True)

    sec("High Risk Records Table", f"threshold ≥ {thresh}")
    high_risk = (pred_df[pred_df["Risk"]=="HIGH"]
                 [["YEAR","MONTH","SUPPLIER","ITEM CODE","ITEM DESCRIPTION",
                   "ITEM TYPE","WAREHOUSE SALES","Prob_High_Demand","Predicted_Class"]]
                 .sort_values("Prob_High_Demand", ascending=False)
                 .head(200))
    st.dataframe(high_risk.style.background_gradient(
        subset=["Prob_High_Demand"], cmap="Blues"),
        use_container_width=True, hide_index=True, height=380)

    csv_pred = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download All Predictions (CSV)", csv_pred,
                       "predictions_2020.csv", "text/csv")


# ══════════════════════════════════════════════════════════
# TAB 6 — PRESCRIPTIVE STRATEGY
# ══════════════════════════════════════════════════════════
with tab6:

    sec("Resource-Constrained Priority Engine", f"Top {capacity} records")
    st.markdown(f"""
    <div class="insight-card">
      <span class="tag">Operational Rule</span><br>
      The inventory team has capacity to review <b>{capacity} item-month records per planning cycle</b>.
      Records are ranked by predicted high-demand probability (desc). Only the top {capacity}
      are surfaced for priority pre-stocking action.
      Adjust the <b>Review Capacity</b> slider in the sidebar to simulate scenarios.
    </div>
    """, unsafe_allow_html=True)

    pred_df2  = ml["pred_df"].copy()
    pred_df2["Action"] = pred_df2["Predicted_Class"].map({
        1: "PRE-STOCK",
        0: "MONITOR",
    })
    queue = (pred_df2[["YEAR","MONTH","SUPPLIER","ITEM CODE","ITEM DESCRIPTION",
                        "ITEM TYPE","WAREHOUSE SALES","Prob_High_Demand",
                        "Predicted_Class","Action"]]
             .sort_values("Prob_High_Demand", ascending=False)
             .reset_index(drop=True)
             .head(capacity))
    queue.index += 1
    queue.index.name = "Rank"

    p1, p2, p3 = st.columns(3)
    pre  = (queue["Predicted_Class"]==1).sum()
    mon  = (queue["Predicted_Class"]==0).sum()
    avg_p = queue["Prob_High_Demand"].mean()
    for col, lbl, val, cls in [
        (p1,"Pre-Stock Items",f"{pre:,}","kpi-pink"),
        (p2,"Monitor Items",  f"{mon:,}","kpi-teal"),
        (p3,"Avg Probability",f"{avg_p:.3f}",""),
    ]:
        col.markdown(f"""
        <div class="kpi-card {cls}" style="min-height:80px;">
          <div class="kpi-title">{lbl}</div>
          <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Waterfall of top-10 queue ─────────────────────────
    sec("Top 20 Priority Items", "Ranked by Probability")
    top20 = queue.head(20).reset_index()
    fig_q = fig_base(height=320)
    colors_q = [C[2] if r==1 else C[0] for r in top20["Predicted_Class"]]
    fig_q.add_trace(go.Bar(
        x=top20["Rank"],
        y=top20["Prob_High_Demand"],
        marker=dict(color=colors_q, line=dict(color=BG, width=1)),
        text=top20["Prob_High_Demand"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        textfont=dict(size=9, color="#94a3b8"),
        customdata=top20[["SUPPLIER","ITEM TYPE"]].values,
        hovertemplate="Rank %{x}<br>Supplier: %{customdata[0]}<br>Type: %{customdata[1]}<br>Prob: %{y:.4f}<extra></extra>",
        name="Priority Score",
    ))
    fig_q.add_hline(y=0.5, line_width=1.5, line_color=C[4],
                     line_dash="dot",
                     annotation_text="Threshold 0.50",
                     annotation_font=dict(color=C[4], size=10))
    fig_q.update_layout(xaxis_title="Priority Rank", yaxis_title="High-Demand Probability",
                         margin=dict(t=20,b=40,l=60,r=20))
    st.plotly_chart(fig_q, use_container_width=True)

    sec("Full Priority Queue", f"{capacity} records")
    st.dataframe(queue.style.background_gradient(
        subset=["Prob_High_Demand"], cmap="Blues"),
        use_container_width=True, height=380)
    csv_q = queue.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 Download Priority Queue CSV (top {capacity})",
                       csv_q, f"priority_queue_top{capacity}.csv", "text/csv")

    st.markdown("---")
    sec("5 Operational Recommendations", "Data-Driven")
    recs = [
        (C[0],"🏭","Pre-stock Wine & Liquor before Q2–Q3",
         "Wine (60%) and Liquor (21%) dominate warehouse volume with a mid-2019 peak. "
         "Increase intake quotas by 20–30% from May onward to absorb seasonal demand."),
        (C[2],"⚠️","Mitigate Top-5 Supplier Concentration Risk",
         "Top 5 suppliers account for ~35% of all warehouse units. "
         "A single supplier disruption can drop throughput 7%+ immediately. Establish backup agreements."),
        (C[1],"🤖","Automate Reorder Alerts at Probability > 0.80",
         "Items scoring ≥ 0.80 should trigger automatic purchase orders. "
         "Items 0.50–0.79 enter the manual review queue. Items < 0.50 follow standard schedules."),
        (C[3],"🔀","Maintain Separate Retail & Warehouse Forecasts",
         "Retail-warehouse sales correlation is only ~0.15. "
         "A shared forecast model would introduce significant noise for both channels."),
        (C[4],"🔄","Annual Model Retraining + Quarterly AUC Monitoring",
         "Retrain the Random Forest annually as new yearly data arrives. "
         "Monitor ROC-AUC quarterly; trigger retraining if it drops below 0.85."),
    ]
    for color, icon, title, body in recs:
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:{color};margin-bottom:10px;">
          <div style="font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">
            {icon} {title}
          </div>
          <div style="font-size:13px;color:#94a3b8;">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e3a5f;margin-top:30px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:11px;color:#334155;padding:10px 0;">
      Warehouse & Retail Sales Intelligence &nbsp;·&nbsp;
      Sowmiya Subramaniyan &nbsp;·&nbsp;
      4-Tier Analytics Platform &nbsp;·&nbsp;
      ROC-AUC 0.9056 · Recall 0.8112
    </div>""", unsafe_allow_html=True)
