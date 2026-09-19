# Warehouse & Retail Sales Intelligence

## Project Description

A production-ready 4-Tier Data Analytics and Machine Learning project built on the
**Warehouse & Retail Sales dataset** (307,645 records, 2017–2020).  
The project demonstrates an end-to-end analytical workflow: data hygiene → EDA →
predictive ML → prescriptive business strategy, with a fully interactive Streamlit dashboard.

---

## Business Problem

Alcohol distributors and warehouse managers need to identify which item-supplier combinations
are likely to experience **high warehouse demand** in advance, so that stock can be pre-positioned
before shortages occur. Manual review of 300K+ records per planning cycle is infeasible.

This project builds a **Random Forest classifier** that scores every item-month combination by
predicted high-demand probability, enabling a capacity-constrained priority queue for inventory teams.

---

## Dataset

| Property       | Value                              |
|----------------|------------------------------------|
| Filename       | `Warehouse_and_Retail_Sales.csv`   |
| Rows           | 307,645                            |
| Columns        | 9                                  |
| Date Range     | January 2017 – December 2020       |
| Source         | Montgomery County (Maryland, USA) Open Data — Alcohol Warehouse & Retail Sales |

**Columns:**  
`YEAR`, `MONTH`, `SUPPLIER`, `ITEM CODE`, `ITEM DESCRIPTION`, `ITEM TYPE`,  
`RETAIL SALES`, `RETAIL TRANSFERS`, `WAREHOUSE SALES`

---

## Technologies Used

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| Data         | pandas, numpy                           |
| Visualisation| matplotlib, seaborn, plotly             |
| ML           | scikit-learn (Random Forest, Pipeline)  |
| Dashboard    | Streamlit                               |
| Reports      | python-docx, nbformat                   |
| Runtime      | Python 3.10+                            |

---

## Project Architecture

```
Warehouse_and_Retail_Sales.csv          ← Source data
Sowmiya_WarehouseRetailSales.py         ← Standalone 4-tier pipeline
Sowmiya_WarehouseRetailSales.ipynb      ← Complete Jupyter notebook
app.py                                  ← Streamlit dashboard
.streamlit/config.toml                  ← Dark analytics theme
requirements.txt                        ← Dependencies
figures/                                ← EDA + ML charts (7 PNGs)
README.md                               ← This file
.gitignore                              ← Git exclusions
Sowmiya_ProjectReport.docx              ← Professional report
```

---

## 4-Tier Methodology

### Tier 1 — Data Hygiene & Architecture
- 307,645 rows loaded; 0 duplicates
- Missing: 167 SUPPLIER, 1 ITEM TYPE, 3 RETAIL SALES → filled with defaults
- Negative values (116 RETAIL SALES, 1,016 RETAIL TRANSFERS, 716 WAREHOUSE SALES) → clipped to 0
- ITEM CODE cast to string; minor ITEM TYPE categories grouped into "OTHER"
- DATE column derived; TOTAL MOVEMENT engineered

### Tier 2 — EDA (5 Visualisations)
1. Monthly Sales Trend (2017–2020)
2. Item Type Distribution (Warehouse vs Retail)
3. Top 15 Suppliers by Warehouse Volume
4. Pearson Correlation Heatmap
5. Item Type × Year Demand Heatmap

### Tier 3 — Predictive ML
- **Derived target:** `High_Warehouse_Demand` = 1 if WAREHOUSE SALES ≥ p90 (10% positive rate)
- **Leakage prevention:** WAREHOUSE SALES, TOTAL MOVEMENT, RETAIL SALES, RETAIL TRANSFERS, ITEM CODE, ITEM DESCRIPTION excluded
- **Model:** Random Forest Classifier
- **Split:** Chronological — 2017-2019 train / 2020 test

### Tier 4 — Prescriptive Strategy
- Capacity-constrained priority queue ranked by predicted probability
- Configurable review capacity: 100 → 10,000 records
- 5 concrete operational recommendations

---

## Data Cleaning

| Problem | Action | Reason |
|---------|--------|--------|
| 167 NULL SUPPLIER | Fill "UNKNOWN SUPPLIER" | Preserve records |
| 1 NULL ITEM TYPE | Fill "UNKNOWN" | Single row |
| 3 NULL RETAIL SALES | Fill 0.0 | Zero movement |
| Negative WH/Retail values | Clip to 0 | Returns/adjustments |
| ITEM CODE mixed types | Cast to str | Identifier |

---

## Model Metrics (Real — computed on 2020 test set)

| Metric     | Score  |
|------------|--------|
| Accuracy   | 0.8672 |
| Precision  | 0.4522 |
| Recall     | 0.8112 |
| F1 Score   | 0.5807 |
| ROC-AUC    | 0.9056 |

---

## Leakage Prevention

Features excluded from ML:
- `WAREHOUSE SALES` — direct target variable
- `TOTAL MOVEMENT` — derived from target
- `RETAIL SALES` — contemporaneous outcome variable
- `RETAIL TRANSFERS` — contemporaneous outcome variable
- `ITEM CODE` / `ITEM DESCRIPTION` — high-cardinality identifiers

Features used: `YEAR`, `MONTH`, `ITEM TYPE` (encoded), `SUPPLIER` (encoded)

---

## Prescriptive Strategy

1. Pre-stock Wine & Liquor before mid-year peak
2. Mitigate top-5 supplier concentration risk
3. Increase Q2–Q3 warehouse intake by 20–30%
4. Maintain separate retail vs warehouse forecasting models
5. Auto-trigger reorder alerts at probability > 0.80

---

## Dashboard Features

- Dark executive analytics theme
- 6 interactive tabs (Overview / Data Quality / EDA / ML / Risk / Strategy)
- Sidebar filters: Year, Month, Item Type
- KPI cards: Warehouse Sales, Retail Sales, Transfers, Products, Suppliers, Monthly Avg
- Interactive Plotly charts throughout
- Capacity-constrained priority queue engine
- CSV download: predictions + priority queue + filtered data

---

## Installation

```bash
git clone <repository-url>
cd warehouse-retail-sales-intelligence

# Create virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Run Pipeline (standalone)

```bash
python Sowmiya_WarehouseRetailSales.py
```

---

## Run Dashboard

```bash
streamlit run app.py
```

Local URL: http://localhost:8501

---

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repository** (public or private)
2. **Upload all project files** to the repository root:
   - `app.py`
   - `Warehouse_and_Retail_Sales.csv`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
4. Click **"New app"**
5. Select your **repository**
6. Select **branch** (usually `main`)
7. Set **Main file path** to `app.py`
8. Click **"Deploy!"**
9. Streamlit will build and deploy; you will receive a public URL like:  
   `https://<your-app-name>.streamlit.app`

> **Note:** The project is deployment-ready, but a public URL must be generated
> after deployment to Streamlit Community Cloud.

---

## Author

**Sowmiya Subramaniyan**  
Data Analytics & Machine Learning Project  
Dataset: Montgomery County Alcohol Warehouse & Retail Sales (2017–2020)
