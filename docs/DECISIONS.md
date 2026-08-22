# FactoryLens — Decision Log

## Decision 1: Dataset Selection
**What:** AI4I 2020 Predictive Maintenance Dataset (UCI)
**Why:** This dataset was created by scientists who replicated 
real sensor patterns from an actual milling machine — making it 
a credible simulation of industrial data. It covers 5 key 
mechanical parameters I understand from my engineering 
background: temperature, rotational speed, torque, and tool 
wear. That domain familiarity meant I could make better 
analytical decisions than someone approaching it purely as 
a data problem.
**Alternative considered:** Generic Kaggle datasets
**Why rejected:** Most are overused in student portfolios 
and lack mechanical domain relevance.

---

## Decision 2: Operator Experience Probabilities
**What:** Junior / Senior / Expert linked to shift 
probabilities — not purely random
**Assumptions:**
- Morning: 20% Junior, 40% Senior, 40% Expert
- Evening: 34% Junior, 33% Senior, 33% Expert
- Night:   50% Junior, 30% Senior, 20% Expert
**Why not pure random:** Pure random assignment would have 
no logic behind it and wouldn't replicate what actually 
happens in real factories. Senior and expert operators 
work day shifts — that is standard across most industries. 
Overnight shifts are handled more by junior staff.
**Validated by:** Crosstab confirmed Morning has highest 
Expert count (1350), Night has highest Junior count (1731).

---

## Decision 3: Machine Age Category Thresholds
**What:** New / Mid-life / Aging / Critical from tool_wear_min
**Thresholds:** 0-100 / 101-160 / 161-200 / 200+
**Why:** Based on standard machining practice. Tools operate 
in primary working life up to 100 minutes. Wear accumulates 
between 101-160. Inspection triggered around 160 minutes 
in real factory schedules. Beyond 200 is past rated life — 
confirmed by failures clustering heavily in this zone.

---

## Decision 4: Maintenance Status Thresholds
**What:** OK / Due Soon / Overdue from tool_wear_min
**Thresholds:** Below 150 → OK, 150-200 → Due Soon, 
Above 200 → Overdue
**Why:** Converts raw sensor reading into actionable 
business alert. A maintenance manager reads Overdue 
and acts — they don't interpret raw numbers.
**Finding:** 30.3% of operations in elevated 
maintenance risk zone.

---

## Decision 5: Business Operations Layer
**What:** Added 4 simulated operational columns —
shift, operator_experience, machine_age_category, 
maintenance_status
**Why:** Original dataset had only sensor readings. 
Real factory analytics needs operational context.
**Transparency:** All simulated variables clearly 
documented. In real deployment these come from 
factory ERP and scheduling systems.

## Decision 6: Model Selection
Models compared: Logistic Regression, Decision Tree, Random Forest
Evaluation metric focus: Recall and F1 — not accuracy

Results (Failure class):
- Logistic Regression: Precision 0.18, Recall 0.85, F1 0.30
- Decision Tree: Precision 0.82, Recall 0.72, F1 0.77
- Random Forest: Precision 0.93, Recall 0.62, F1 0.74

Winner: Decision Tree
Reason: Best F1 score (0.77) with strongest balance between 
Precision and Recall. In manufacturing, missing a failure is 
more costly than a false alarm — making balanced Recall 
critical. Random Forest retained for SHAP explainability.


# Findings

## Finding 1: Tool Wear and Failure Relationship
Median tool wear for failed machines: 165 minutes
Median tool wear for non-failed machines: 107 minutes
Conclusion: Tool wear is a strong predictor of failure risk.
Validates machine_age_category Critical threshold at 200 minutes.

## Finding 2: Product Type and Failure Rate
L type: 3.92% | M type: 2.77% | H type: 2.09%
L type fails at nearly double the rate of H type.
Used failure rate not raw count — L type is 60% of production
volume so raw counts would be misleading.
Mechanical explanation: Lower quality variants put more stress
on cutting tools during processing.

## Finding 3: Operator Experience and Failure Rate
Junior: 3.75% | Expert: 3.47% | Senior: 2.95%
Junior highest — confirmed design assumption.
Unexpected: Expert higher than Senior.
Interpretation: Expert operators handle more demanding tasks,
carrying higher inherent failure risk despite greater skill.
Conclusion: Experience matters but is not the sole failure driver.

## Finding 4: Temperature Difference and Failure Risk
No Failure median temp_diff: 9.8K
Failure median temp_diff: 9.3K
Counterintuitive finding: Failed machines have LOWER temp difference.
Explanation: Low temperature gradient indicates poor heat dissipation
— the machine cannot push heat outward effectively.
Aligns with Heat Dissipation Failure (HDF) type in dataset.
Feature temp_diff confirmed as valuable for modeling.
 
## Finding 5: Machine Age Category and Failure Rate
New: 2.25% | Mid-life: 2.11% | Aging: 3.18% | Critical: 15.49%
Critical category fails at 5x the rate of all other categories.
This validates the 200 minute Critical threshold decision.
Key business recommendation: machines crossing 200 minutes tool
wear require immediate maintenance intervention.
This is the strongest single finding in the EDA.


## Finding 6: Random Forest Confusion Matrix Results
Test set: 2000 rows, 68 real failures
True Positives (caught): 42 (61.8%)
False Negatives (missed): 26 (38.2%)
False Positives (false alarms): 3
True Negatives: 1929

Strength: Only 3 false alarms — high operational credibility
Limitation: 38.2% of failures missed
Mitigation: Rule-based maintenance_status alerts provide 
additional safety net beyond model predictions.

## Finding 7: SHAP Feature Importance
Top 5 features driving failure predictions:
1. rotational_speed_rpm  (0.1051)
2. power_kw — engineered feature (0.1046)
3. torque_nm (0.0920)
4. tool_wear_min (0.0659)
5. temp_diff — engineered feature (0.0614)

Key insight: Two of top 5 are engineered features created 
from mechanical domain knowledge. power_kw ranked 2nd out 
of 17 features — validates feature engineering decision.

Model relies primarily on machine workload intensity 
(speed, power, torque) over wear metrics alone.
Operator experience and shift ranked lowest — consistent 
with EDA finding that experience alone doesn't drive failures.