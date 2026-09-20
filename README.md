# HealthConnect Clinic — Appointment No-Show Analysis

**AnalystLab Africa Experience Lab | Data Analytics Track — Week 7**
*(Continuity project — builds on Weeks 4-6: problem understanding, analysis, advanced validation)*

## 📌 Project Overview

HealthConnect Clinic is a fictional healthcare provider facing a significant operational challenge: a high rate of missed appointments (no-shows), which leads to inefficient use of appointment slots, repetitive administrative burden, and reduced quality of patient care.

**Central Project Question:** *How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?*

This is a shared, multi-track Experience Lab project. The project progresses week by week:

| Week | Stage |
|---|---|
| Week 4 | Problem Understanding → Resource Review → Solution Planning |
| Week 5 | Analysis → Development → Initial Implementation |
| Week 6 | Integration → Advanced Development → Validation |
| **Week 7** | **Testing → Refinement → End-to-End Validation** |
| Week 8 | Final Integration → Presentation |

## 🗂️ Repository Structure

```
├── week4-kickoff/
│   ├── Initial_Analysis_Document.docx
│   └── Week4_Project_Summary.docx
├── week5-analysis/
│   ├── HealthConnect_Appointment_Data_cleaned.csv
│   ├── Initial_HealthConnect_Analytics_Report.docx
│   └── Week5_Project_Summary.docx
├── week6-advanced-analytics/
│   ├── HealthConnect_Advanced_Analytics_Report.docx
│   ├── HealthConnect_Feature_Validation_Model.py
│   └── Week6_Project_Summary.docx
├── week7-testing-refinement/
│   ├── HealthConnect_Analytics_Testing_Refinement_Report.docx   # Test log, statistical audit, threshold refinement
│   ├── HealthConnect_Model_Testing_Refinement.py                # Reproducible test → finding → action → retest cycle
│   └── Week7_Project_Summary.docx
└── README.md
```

## 📊 Dataset

**File:** `HealthConnect_Appointment_Data.csv` (5,000 fictional, anonymized appointment records, 18 variables). The original file is never modified; a cleaned copy is maintained separately (`week5-analysis/HealthConnect_Appointment_Data_cleaned.csv`).

## 🔍 Week 7 — What's New (not a repeat of Week 6)

Week 7 does not recompute the analysis. It **tests** what was already built, following a mandatory Test → Finding → Action → Retest cycle:

| Test | Result |
|---|---|
| **KPI accuracy audit** | 13 previously published values (lead time, no-show history, reminder channel) independently recalculated — **100% match**, no calculation errors found |
| **Statistical anomaly investigation** | The counter-intuitive "0-7 day lead time × 3+ prior no-shows" heatmap cell (23%) tested with a Wilson 95% CI — **confirmed as small-sample noise** (n=13, CI = [8.2%, 50.3%]), not a genuine reversal |
| **Cross-track model error analysis (mandatory)** | Week 6 model's errors decomposed by risk segment — revealed the default threshold missed **34.3% of true no-shows** in the "Low Risk" segment despite 0% missed in "High Risk" |
| **Refinement action** | Classification threshold lowered from 0.50 to 0.40, justified by asymmetric business cost (a missed no-show costs more than an unnecessary reminder) |
| **Retest result** | False-negative rate on "Low Risk" dropped from 34.3% to **24.1%**; F1-score improved from 0.652 to **0.686**; "High Risk" segment unaffected (still 0% missed) |

**Headline insight:** A model that looked solid on aggregate accuracy (62.4%) was silently failing on over a third of no-shows in its "safest" segment. Testing by business segment — not just by overall score — is what surfaced this, and the fix was a simple, well-justified threshold adjustment rather than a model rebuild.

## 🔗 Cross-Track Integration (Week 7 — tested, not just built)

Building on the Week 6 integration artefact (a baseline model using 4 Data-Analytics-validated features), Week 7 tested that artefact rather than accepting its aggregate score at face value:

- **Test performed:** error breakdown by the Week 6 risk segmentation (Low / Medium / Medium / High Risk)
- **Finding:** severe imbalance in false negatives across segments (34.3% vs. 0%)
- **Action:** threshold recalibration (0.50 → 0.40)
- **Retest:** confirmed improvement without harming the "High Risk" segment
- **Evidence:** `HealthConnect_Model_Testing_Refinement.py` (fully reproducible, prints before/after comparison)

## ✅ Recommendations (Week 7, refined from Week 6)

1. Adopt a 0.40 classification threshold (not 0.50) for any operational use of the no-show prediction model.
2. Do not base business rules on heatmap cells with fewer than 30 observations (now clearly flagged in the refined visualization).
3. Maintain standard reminders even for "Low Risk" patients — this segment still carries a non-trivial 34.5% no-show rate, and even the refined model still misses roughly a quarter of them.

Full details, statistical tests, and before/after visualizations are available in `week7-testing-refinement/HealthConnect_Analytics_Testing_Refinement_Report.docx`.

## 🛠️ Tools & Techniques

- Python (pandas, matplotlib, scipy) for statistical validation and hypothesis testing
- Wilson confidence interval and binomial testing for small-sample anomaly investigation
- scikit-learn for model error analysis and threshold-based refinement

## ▶️ Next Steps (Week 8)

- Consolidate Weeks 4-7 into a coherent final narrative for the project presentation.
- Validate the operational feasibility of the 0.40 threshold recommendation from a Project Management perspective.
- Present the full validation → test → refinement cycle as a demonstration of analytical rigor.

## 👤 Author

Josfrid AGBADOGBE, Data Analytics Intern, AnalystLab Africa Experience Lab
*Week 7 — HealthConnect Testing, Refinement & End-to-End Validation*
