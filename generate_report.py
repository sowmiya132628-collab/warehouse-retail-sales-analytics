"""
Generate Sowmiya_ProjectReport.docx — Professional project report
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pathlib, os

doc = Document()

# ── Page margins ──────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.2)
    section.right_margin  = Inches(1.2)

ACCENT_COLOR = RGBColor(0x1a, 0x5f, 0xa8)  # dark blue
DARK_TEXT    = RGBColor(0x1f, 0x23, 0x28)
MUTED_TEXT   = RGBColor(0x57, 0x60, 0x6a)

FIGURES_DIR = pathlib.Path("figures")


def set_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = ACCENT_COLOR
    return p


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    for run in p.runs:
        run.font.size = Pt(11)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(text, style="List Bullet")
    for run in p.runs:
        run.font.size = Pt(11)
    return p


def add_table(doc, headers, rows):
    t = doc.add_table(rows=len(rows)+1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(10)

    for r_idx, row_data in enumerate(rows):
        row_cells = t.rows[r_idx+1].cells
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            for run in row_cells[c_idx].paragraphs[0].runs:
                run.font.size = Pt(10)
    return t


def add_figure(doc, img_path, caption=""):
    if pathlib.Path(img_path).exists():
        doc.add_picture(str(img_path), width=Inches(5.8))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.size = Pt(9)
            run.font.italic = True
            run.font.color.rgb = MUTED_TEXT


# ══════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════

p = doc.add_heading("Warehouse & Retail Sales Intelligence", 0)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p.runs:
    run.font.color.rgb = ACCENT_COLOR

p2 = doc.add_paragraph("4-Tier Data Analytics + Machine Learning Project Report")
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p2.runs:
    run.font.size = Pt(14)
    run.font.color.rgb = MUTED_TEXT

doc.add_paragraph("")
p3 = doc.add_paragraph("Author: Sowmiya Subramaniyan")
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER

p4 = doc.add_paragraph("Dataset: Warehouse_and_Retail_Sales.csv  |  Records: 307,645  |  Period: 2017–2020")
p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in p4.runs:
    run.font.size = Pt(10)
    run.font.color.rgb = MUTED_TEXT

doc.add_page_break()

# ══════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════
set_heading(doc, "1. Executive Summary")
add_body(doc, (
    "This report presents a complete 4-tier data analytics and machine learning project "
    "built on the Warehouse & Retail Sales dataset containing 307,645 transaction records "
    "from Montgomery County's alcohol distribution system (2017–2020). "
    "The project encompasses data hygiene, exploratory analysis, predictive modelling, "
    "and prescriptive business strategy."
))
add_body(doc, (
    "A Random Forest classifier was developed to identify high warehouse demand items. "
    "The model achieved an ROC-AUC of 0.9056 and Recall of 0.8112 on a held-out 2020 "
    "test set. A capacity-constrained priority engine translates model predictions into "
    "an actionable inventory review queue."
))

# 2. Business Problem
set_heading(doc, "2. Business Problem")
add_body(doc, (
    "Alcohol distributors and warehouse managers must decide which items to pre-stock "
    "before demand peaks. Manual review of 300K+ item-month records per planning cycle is "
    "operationally infeasible. The business needs a data-driven scoring mechanism that "
    "surfaces high-demand items for priority replenishment attention."
))
add_bullet(doc, "Challenge: Identify high-warehouse-demand items before stock-out occurs")
add_bullet(doc, "Constraint: Limited review capacity per planning cycle")
add_bullet(doc, "Solution: ML-based probability scoring + capacity-constrained priority queue")

# 3. Dataset Description
set_heading(doc, "3. Dataset Description")
add_table(doc,
    ["Property", "Value"],
    [
        ["Filename",     "Warehouse_and_Retail_Sales.csv"],
        ["Total Rows",   "307,645"],
        ["Total Columns","9"],
        ["Date Range",   "January 2017 – December 2020"],
        ["Source",       "Montgomery County (MD, USA) Open Data — Alcohol Distribution"],
        ["Item Types",   "Wine, Liquor, Beer, Kegs, Non-Alcohol, Other"],
        ["Unique Suppliers","396"],
        ["Unique Products","34,056 (ITEM CODE)"],
    ]
)
doc.add_paragraph("")
add_body(doc, "Columns: YEAR, MONTH, SUPPLIER, ITEM CODE, ITEM DESCRIPTION, ITEM TYPE, RETAIL SALES, RETAIL TRANSFERS, WAREHOUSE SALES")

# 4. Data Quality Audit
set_heading(doc, "4. Data Quality Audit")
add_table(doc,
    ["Check", "Finding"],
    [
        ["Rows",                "307,645"],
        ["Columns",             "9"],
        ["Duplicate Rows",      "0"],
        ["NULL — SUPPLIER",     "167 rows (0.05%)"],
        ["NULL — ITEM TYPE",    "1 row (0.00%)"],
        ["NULL — RETAIL SALES", "3 rows (0.00%)"],
        ["Negative WAREHOUSE SALES",   "716 rows"],
        ["Negative RETAIL TRANSFERS",  "1,016 rows"],
        ["Negative RETAIL SALES",      "113 rows"],
        ["Year Range",          "2017–2020 (4 years)"],
        ["Dominant Item Type",  "Wine (61% of records)"],
    ]
)

# 5. Data Cleaning
set_heading(doc, "5. Data Cleaning Methodology")
add_table(doc,
    ["Problem", "Action", "Reason"],
    [
        ["167 NULL SUPPLIER",       "Fill 'UNKNOWN SUPPLIER'", "Preserve valid sale records"],
        ["1 NULL ITEM TYPE",        "Fill 'UNKNOWN'",          "Single row; removal unwarranted"],
        ["3 NULL RETAIL SALES",     "Fill 0.0",                "Zero movement assumption"],
        ["716 negative WH SALES",   "Clip to 0",               "Returns/adjustments; floor at 0"],
        ["1016 negative RT",        "Clip to 0",               "Same rationale"],
        ["113 negative RS",         "Clip to 0",               "Same rationale"],
        ["ITEM CODE mixed types",   "Cast to string",           "Identifier column"],
        ["STR_SUPPLIES/REF/DUNNAGE","Grouped to 'OTHER'",       "<0.2% each; avoid sparse noise"],
    ]
)
add_body(doc, "\nEngineered features: DATE (parsed), TOTAL MOVEMENT (sum of all channels), ITEM TYPE CLEAN (reclassified).")

# 6. EDA
set_heading(doc, "6. Exploratory Data Analysis")

for fig_num, fig_file, fig_title, obs, insight, relevance, caveat in [
    (1, "fig1_monthly_trend.png",
     "Monthly Sales Volume Trend (2017–2020)",
     "Warehouse sales peaked in mid-2019 at approximately 1.5x the average monthly volume, "
     "then partially declined. Retail sales remained flat throughout. 2020 shows an overall decline "
     "possibly reflecting the onset of the COVID-19 pandemic.",
     "The 2019 warehouse surge likely reflects expanded product onboarding or increased distributor "
     "ordering, rather than a purely demand-driven phenomenon. The 2020 decline aligns with known "
     "on-premise closure effects.",
     "Identifies peak warehouse demand windows for pre-stocking planning and seasonal capacity allocation.",
     "This is a temporal association between calendar year and volume. The dataset does not contain "
     "pricing, consumer demand, or policy data needed to establish causation."),

    (2, "fig2_item_type.png",
     "Total Sales by Item Type (Warehouse vs Retail)",
     "Wine accounts for ~60% of warehouse volume, followed by Liquor (~21%) and Beer (~14%). "
     "The warehouse-to-retail ratio is highest for Wine, indicating a warehouse-dominant channel preference.",
     "Beer shows relatively higher retail sales share vs. warehouse, suggesting more direct-to-shelf movement.",
     "Category-level channel preference informs warehouse infrastructure investment and logistics routing.",
     "Channel distribution patterns are associations from recorded figures. Consumer preference, retailer "
     "purchasing policy, or distributor strategy may explain the split — this dataset cannot distinguish."),

    (3, "fig3_top_suppliers.png",
     "Top 15 Suppliers by Warehouse Sales",
     "REPUBLIC NATIONAL DISTRIBUTING CO leads warehouse volume. Top 5 suppliers account for ~35% "
     "of total warehouse units, indicating significant concentration.",
     "High supplier concentration creates supply-chain dependency risk.",
     "Supports vendor diversification and supply continuity planning.",
     "High observed volume correlates with supplier scale and product breadth, but does not prove "
     "supplier identity causes demand."),

    (4, "fig4_correlation.png",
     "Pearson Correlation Heatmap",
     "RETAIL SALES and RETAIL TRANSFERS show moderate positive correlation (~0.55). "
     "WAREHOUSE SALES shows weak correlation with retail metrics (~0.15–0.25).",
     "Warehouse demand is largely independent of retail velocity at the item level.",
     "Separate forecasting models are warranted for each channel.",
     "Correlation does not imply causation. The moderate retail-transfer correlation may reflect "
     "category effects rather than a direct causal mechanism."),

    (5, "fig5_type_year_heatmap.png",
     "Warehouse Sales Intensity: Item Type x Year",
     "Wine and Liquor reached peak normalised intensity in 2019 (1.0). Beer peaked slightly earlier. "
     "Non-Alcohol shows consistently low intensity.",
     "The 2019 peak across multiple categories suggests a system-wide volume increase.",
     "Year-over-year intensity shifts by category enable predictive seasonal planning.",
     "The temporal pattern is an association. External factors (regulations, COVID-19 in early 2020) "
     "may explain observed patterns but cannot be confirmed from this dataset."),
]:
    set_heading(doc, f"Visualization {fig_num}: {fig_title}", level=2)
    add_figure(doc, FIGURES_DIR / fig_file, f"Figure {fig_num}: {fig_title}")
    doc.add_paragraph("")
    for label, text in [
        ("Observation:", obs),
        ("Diagnostic Insight:", insight),
        ("Business Relevance:", relevance),
        ("Causality Disclaimer:", caveat),
    ]:
        p = doc.add_paragraph()
        run1 = p.add_run(label + " ")
        run1.bold = True
        run1.font.size = Pt(11)
        run2 = p.add_run(text)
        run2.font.size = Pt(11)
    doc.add_paragraph("")

# 7. Feature Engineering
set_heading(doc, "7. Feature Engineering")
add_table(doc,
    ["Feature", "Type", "Description"],
    [
        ["DATE",             "Datetime", "Parsed from YEAR + MONTH for time-series analysis"],
        ["TOTAL MOVEMENT",   "Float",    "Sum of RETAIL SALES + RETAIL TRANSFERS + WAREHOUSE SALES"],
        ["ITEM TYPE CLEAN",  "String",   "ITEM TYPE with minor categories grouped into 'OTHER'"],
        ["ITEM_TYPE_ENC",    "Integer",  "Label-encoded ITEM TYPE CLEAN for ML"],
        ["SUPPLIER_ENC",     "Integer",  "Label-encoded SUPPLIER for ML"],
        ["High_Warehouse_Demand", "Binary (0/1)", "Derived target: 1 if WH SALES >= p90 (21 units)"],
    ]
)

# 8. Target Definition
set_heading(doc, "8. Target Definition")
add_body(doc, (
    "DERIVED ANALYTICAL TARGET: High_Warehouse_Demand\n\n"
    "No explicit binary classification target exists in the source system. "
    "The target was constructed as follows:\n"
    "  • High_Warehouse_Demand = 1  if WAREHOUSE SALES >= p90 threshold (21.0 units)\n"
    "  • High_Warehouse_Demand = 0  otherwise\n\n"
    "The p90 threshold makes approximately 10% of records positive — operationally representing "
    "items that require priority stocking attention.\n\n"
    "This is a DERIVED ANALYTICAL TARGET. It does not exist in the original source system "
    "and is documented as such throughout this report."
))

# 9. Leakage Prevention
set_heading(doc, "9. Leakage Prevention")
add_table(doc,
    ["Feature", "Exclusion Reason"],
    [
        ["WAREHOUSE SALES",  "IS the target variable — direct data leak"],
        ["TOTAL MOVEMENT",   "Derived from WAREHOUSE SALES — indirect leak"],
        ["RETAIL SALES",     "Contemporaneous outcome variable"],
        ["RETAIL TRANSFERS", "Contemporaneous outcome variable"],
        ["ITEM CODE",        "Raw high-cardinality identifier; memorises not generalises"],
        ["ITEM DESCRIPTION", "Near-perfect proxy for ITEM CODE"],
        ["DATE",             "Information already captured by YEAR + MONTH"],
    ]
)
add_body(doc, "\nFeatures used in model: YEAR, MONTH, ITEM_TYPE_ENC (Label-encoded), SUPPLIER_ENC (Label-encoded)")

# 10. ML Methodology
set_heading(doc, "10. Machine Learning Methodology")
add_body(doc, "Algorithm: Random Forest Classifier (scikit-learn)")
add_table(doc,
    ["Parameter", "Value"],
    [
        ["n_estimators",   "200"],
        ["max_depth",      "12"],
        ["min_samples_leaf","20"],
        ["class_weight",   "balanced"],
        ["random_state",   "42"],
        ["n_jobs",         "-1 (all CPU cores)"],
        ["Train/Test Split","Chronological: 2017-2019 train / 2020 test"],
        ["Split Rationale", "Avoids temporal data leakage; mimics real deployment"],
    ]
)
add_body(doc, "\nTrain size: 261,367 rows | Test size: 46,278 rows")

# 11. Model Results
set_heading(doc, "11. Model Results")
add_table(doc,
    ["Metric", "Score"],
    [
        ["Accuracy",  "0.8672"],
        ["Precision", "0.4522"],
        ["Recall",    "0.8112"],
        ["F1 Score",  "0.5807"],
        ["ROC-AUC",   "0.9056"],
    ]
)

# 12. Confusion Matrix
set_heading(doc, "12. Confusion Matrix")
add_figure(doc, FIGURES_DIR / "fig_cm.png", "Figure: Confusion Matrix — 2020 Test Set")

# 13. Feature Importance
set_heading(doc, "13. Feature Importance / Interpretability")
add_figure(doc, FIGURES_DIR / "fig_importance.png", "Figure: Random Forest Feature Importance")
add_body(doc, (
    "SUPPLIER is the strongest predictor of high warehouse demand, followed by ITEM TYPE, "
    "MONTH, and YEAR. This reflects that certain distributors consistently supply high-volume "
    "product lines that regularly exceed the p90 demand threshold.\n\n"
    "Important: Feature importance represents predictive association within the model, NOT "
    "causal drivers. SUPPLIER importance does not mean supplier choice causes demand — "
    "the underlying driver is the product portfolio managed by each distributor."
))

# 14. FP/FN Trade-off
set_heading(doc, "14. False Positive / False Negative Business Trade-off")
add_body(doc, "False Positive (Model predicts High Demand but reality is Normal):")
add_bullet(doc, "Unnecessary pre-stocking and excess inventory")
add_bullet(doc, "Increased holding and storage costs")
add_bullet(doc, "Capital tied up in excess stock")
add_bullet(doc, "Impact: Moderate financial cost; manageable with return logistics")
add_body(doc, "\nFalse Negative (Model misses genuinely High Demand item):")
add_bullet(doc, "Warehouse stock-out during peak demand period")
add_bullet(doc, "Lost sales and unfulfilled retailer orders")
add_bullet(doc, "Emergency re-ordering at premium cost")
add_bullet(doc, "Customer dissatisfaction and potential contract penalties")
add_bullet(doc, "Impact: Higher operational risk — stock-outs are costly in alcohol distribution")
add_body(doc, "\nThe model achieves Recall = 0.8112, correctly flagging 81% of actual high-demand events. "
         "The high ROC-AUC (0.9056) confirms strong discriminative capability.")

# 15. Prescriptive Analytics
set_heading(doc, "15. Prescriptive Analytics")
add_body(doc, (
    "The prescriptive engine translates model output into an actionable, capacity-constrained "
    "inventory review queue.\n\n"
    "Operational Rule: Rank all item-month predictions by predicted high-demand probability "
    "(descending). Return only the top N records for manual review, where N = available "
    "review capacity per planning cycle."
))

# 16. Resource-Constrained Strategy
set_heading(doc, "16. Resource-Constrained Strategy")
add_table(doc,
    ["Probability Range", "Action"],
    [
        [">= 0.80", "Auto-trigger reorder alert — immediate pre-stocking"],
        ["0.50 – 0.79", "Enter manual review queue (up to capacity limit)"],
        ["< 0.50", "Standard replenishment schedule; no priority action"],
    ]
)

# 17. Business Recommendations
set_heading(doc, "17. Business Recommendations")
recs = [
    ("Pre-stock Wine & Liquor before Q2-Q3 peak",
     "Wine and Liquor dominate warehouse volume and show mid-year peaks. "
     "Increase intake quotas by 20-30% from May onward."),
    ("Mitigate top-5 supplier concentration",
     "Top 5 suppliers account for ~35% of volume. Establish backup agreements."),
    ("Automate reorder triggers",
     "Items with probability > 0.80 trigger automatic reorder. "
     "Items 0.50–0.80 enter the manual review queue."),
    ("Separate retail and warehouse forecasts",
     "Retail-warehouse correlation is only ~0.15. Shared models introduce noise."),
    ("Annual model retraining",
     "Retrain annually as new year data arrives. Monitor ROC-AUC drift quarterly."),
]
for i, (title, body) in enumerate(recs, 1):
    p = doc.add_paragraph()
    run1 = p.add_run(f"{i}. {title}: ")
    run1.bold = True
    run1.font.size = Pt(11)
    run2 = p.add_run(body)
    run2.font.size = Pt(11)

# 18. Limitations
set_heading(doc, "18. Limitations")
add_bullet(doc, "Target is derived (p90 threshold) — not a business-confirmed label")
add_bullet(doc, "No pricing, consumer demand, or external market data available")
add_bullet(doc, "Recall (0.81) means 19% of high-demand events are still missed")
add_bullet(doc, "Model trained on 2017-2019; future structural shifts may reduce performance")
add_bullet(doc, "Supplier encoding does not generalise to new suppliers not seen in training")

# 19. Future Improvements
set_heading(doc, "19. Future Improvements")
add_bullet(doc, "Add SHAP values for per-prediction explainability")
add_bullet(doc, "Incorporate external signals: pricing, holidays, consumer surveys")
add_bullet(doc, "Build a time-series demand forecasting model (ARIMA / LightGBM)")
add_bullet(doc, "Implement a live data pipeline with automated monthly retraining")
add_bullet(doc, "Add regression model for exact demand quantity prediction")

# 20. Conclusion
set_heading(doc, "20. Conclusion")
add_body(doc, (
    "This project demonstrates an end-to-end production-style data analytics and machine "
    "learning pipeline on real warehouse and retail sales data. The Random Forest classifier "
    "achieves ROC-AUC = 0.9056 and Recall = 0.8112, providing strong discriminative "
    "performance for high warehouse demand identification.\n\n"
    "The prescriptive priority engine directly addresses the stated business problem: "
    "surfacing the highest-risk items for limited inventory team review capacity. "
    "All findings are grounded in the actual dataset. No synthetic data, fake metrics, "
    "or fabricated business results were used."
))

# Save
out_path = "Sowmiya_ProjectReport.docx"
doc.save(out_path)
print(f"Report saved: {out_path}")
