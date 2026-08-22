# FactoryLens 🏭
### Manufacturing Operations Analytics Platform

Modern manufacturing generates large volumes of operational data, yet production managers often struggle to convert sensor readings into actionable decisions. FactoryLens demonstrates how data analytics and machine learning can support manufacturing operations — identifying failure patterns, explaining risk factors, and presenting insights through an interactive dashboard.

My mechanical engineering background provided operational context. The project focuses on applying data science to solve a real business problem.

---

## Business Problem

A manufacturing facility running CNC milling machines needs to answer four operational questions:

- Which shifts and product types produce the most failures?
- Which machine conditions are early warning signs?
- Can we predict failure before it happens?
- What should the maintenance team do about it?

FactoryLens answers all four.

---

## Dataset

**Source:** AI4I 2020 Predictive Maintenance Dataset
**Repository:** UCI Machine Learning Repository
**Size:** 10,000 observations, 14 original features
**Domain:** CNC milling machine sensor readings

The base dataset was enriched with four simulated operational variables to reflect real factory scheduling context:

| Variable | Description | Method |
|---|---|---|
| `shift` | Morning / Evening / Night | Cyclic assignment |
| `operator_experience` | Junior / Senior / Expert | Shift-linked weighted probability |
| `machine_age_category` | New / Mid-life / Aging / Critical | Derived from tool wear thresholds |
| `maintenance_status` | OK / Due Soon / Overdue | Rule-based from tool wear |

> These variables are simulated and clearly documented. In real deployment they would come from factory ERP and scheduling systems.

---

## Methodology
Raw Sensor Data
↓
Data Cleaning & Validation
↓
Business Layer Enrichment (4 operational variables)
↓
Exploratory Data Analysis (6 business questions)
↓
Feature Engineering (3 mechanical features)
↓
Model Comparison (Logistic Regression, Decision Tree, Random Forest)
↓
SHAP Explainability
↓
Interactive Streamlit Dashboard

---

## Key Findings

| Finding | Insight |
|---|---|
| Shift Analysis | Evening shift has highest failure rate (3.66%) |
| Tool Wear | Failed machines have median tool wear 58 min higher |
| Product Type | L type fails at 3.92% — nearly double H type (2.09%) |
| Operator Experience | Junior operators show highest failure rate (3.75%) |
| Temperature | Lower temp difference correlates with failure risk |
| Machine Age | Critical machines fail at 15.49% — 5x other categories |

---

## Feature Engineering

Three features derived from mechanical domain knowledge:

| Feature | Formula | Rationale |
|---|---|---|
| `power_kw` | Torque × RPM / 9549 | Mechanical power — high power = high stress |
| `temp_diff` | Process temp − Air temp | Thermal stress indicator |
| `tool_wear_category_num` | Ordered encoding 0-3 | Converts wear to risk progression |

**SHAP Result:** `power_kw` ranked 2nd of 17 features — validating the feature engineering decision.

---

## Model Performance

Three models compared on Failure class metrics:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.18 | 0.85 | 0.30 |
| Decision Tree | 0.82 | 0.72 | 0.77 |
| **Random Forest** | **0.93** | **0.62** | **0.74** |

**Selected: Random Forest**
Reason: Highest Precision (0.93) — only 3 false alarms out of 2000 test cases. High precision maintains operational credibility — maintenance teams trust systems that don't cry wolf.

**Confusion Matrix (Random Forest):**
- Real failures in test set: 68
- Caught: 42 (61.8%)
- Missed: 26 (38.2%)
- False alarms: 3

---

## Dashboard

Three sections built in Streamlit:

**1. Operations Overview**
Four KPI cards — failure rate, total observations, highest risk shift, critical machine percentage.

**2. Business Insights Explorer**
Interactive dropdown — select any of 6 business questions, chart renders instantly.

**3. Failure Risk Predictor**
Input machine parameters via sliders → failure probability → risk level (High/Medium/Low) → maintenance recommendation.

---

## Current Limitations

- Operational variables are simulated — not from real factory scheduling systems
- Dataset represents one manufacturing scenario (CNC milling machine)
- Model catches 61.8% of failures — real deployment would require threshold tuning and more failure examples
- Real deployment would require live sensor integration

---

## Future Scope

- Live IoT sensor data integration
- SQL database backend for production logging
- Automated SMS/email alerts on high-risk predictions
- Power BI executive dashboard
- Cloud deployment on AWS or Azure

---

## Project Structure
FactoryLens/
│
├── data/
│   ├── ai4i2020.csv                    ← original dataset
│   └── factorylens_dataset.csv         ← enriched dataset
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_business_layer_and_eda.ipynb
│   └── 03_feature_engineering_and_ml.ipynb
│
├── streamlit_app.py
│
├── docs/
│   ├── Decision_Log.md
│   └── Business_Questions.md
│
├── requirements.txt
└── README.md

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/FactoryLens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run streamlit_app.py
```

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · SHAP · Matplotlib · Seaborn · Streamlit

---

*Built by Yash Patel
*Dataset: UCI Machine Learning Repository — AI4I 2020*