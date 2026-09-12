# HealthConnect Clinic — Appointment No-Show Analysis

**AnalystLab Africa Experience Lab | Data Analytics Track — Week 6**
*(Continuity project — builds on Week 4 problem understanding and Week 5 initial analysis)*

## 📌 Project Overview

HealthConnect Clinic is a fictional healthcare provider facing a significant operational challenge: a high rate of missed appointments (no-shows), which leads to inefficient use of appointment slots, repetitive administrative burden, and reduced quality of patient care.

**Central Project Question:** *How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?*

This is a shared, multi-track Experience Lab project. The project progresses week by week:

| Week | Stage |
|---|---|
| Week 4 | Problem Understanding → Resource Review → Solution Planning |
| Week 5 | Analysis → Development → Initial Implementation |
| **Week 6** | **Integration → Advanced Development → Validation** |
| Week 7 | Testing → Refinement → End-to-End Validation |
| Week 8 | Final Integration → Presentation |

## 🗂️ Repository Structure

```
├── week4-kickoff/
│   ├── Initial_Analysis_Document.pdf
│   └── Week4_Project_Summary.pdf
├── week5-analysis/
│   ├── HealthConnect_Appointment_Data_cleaned.csv
│   ├── Initial_HealthConnect_Analytics_Report.pdf
│   └── Week5_Project_Summary.pdf
├── week6-advanced-development-validation/
│   ├── HealthConnect_Advanced_Analytics_Report.pdf     # Statistical validation, combined effects, risk segmentation
│   ├── HealthConnect_Feature_Validation_Model.py       # Cross-track integration artefact (predictive validation)
│   └── Week6_Project_Summary.pdf
└── README.md
```

## 📊 Dataset

**File:** `HealthConnect_Appointment_Data.csv` (5,000 fictional, anonymized appointment records, 18 variables). The original file is never modified; a cleaned copy is maintained separately (`week5-analysis/HealthConnect_Appointment_Data_cleaned.csv`).

## 🔍 Week 6 — What's New (not a repeat of Week 5)

Week 6 does not recompute the Week 5 KPIs. It **validates, deepens, and integrates** them:

| Analysis | Result |
|---|---|
| **Statistical validation (Chi² test)** | Booking lead time (p < 0.0001) and prior no-show history (p < 0.0001) confirmed as highly significant — not sampling artifacts |
| **Combined effect** (lead time × no-show history) | No-show rate reaches **91%** for the worst combined profile (46-60 day lead time + 3+ prior no-shows) |
| **Actionable risk segmentation** | 4-tier risk profile built: **"High Risk"** patients (long lead time + prior no-shows) = 977 appointments (20.6% of volume) at **70.5%** no-show, vs. 34.5% for "Low Risk" |
| **Robustness check** | Lead-time effect holds consistently across all 4 appointment types — confirmed as a clinic-wide phenomenon, not a specialty-specific artifact |
| **Cross-track integration (mandatory this week)** | Built and evaluated a baseline logistic regression model using the 4 validated KPIs as features — **62.4% accuracy vs. 51.1% naive baseline** (+11.3 points), confirming genuine predictive value |

## 🔗 Cross-Track Integration (Week 6 requirement)

No live Data Science teammate was available in this solo cohort track. To meet the Week 6 requirement that *"communication alone is not sufficient — integration must produce a meaningful change, improvement, decision, or validated output,"* a genuine **model artefact** was produced instead of a simulated exchange:

- **Input provided:** the 4 KPIs validated by Data Analytics (booking lead time, prior no-show history, distance to clinic, reminder status)
- **Activity performed:** trained and evaluated a logistic regression classifier using exclusively these 4 features
- **Result:** the model outperforms a naive baseline by +11.3 accuracy points (ROC-AUC 0.665), with `booking_lead_days` emerging as the strongest coefficient — an exact match with the descriptive analysis
- **Evidence:** `HealthConnect_Feature_Validation_Model.py` (fully reproducible)
- **Project benefit:** confirms the 4 KPIs are a sound feature foundation for a future production model, saving the Data Science track re-selection work

## ✅ Recommendations (Week 6, refined from Week 5)

1. Deploy a risk-based alert system targeting the 977 "High Risk" appointments (long lead time + prior no-show history) with personalized confirmation calls.
2. Treat booking lead time as a clinic-wide structural issue (confirmed across all appointment types), not a specialty-specific problem.
3. Concentrate administrative follow-up resources on the ~20% of appointments flagged as high-risk, rather than spreading effort evenly across all patients.

Full details, statistical tests, and visualizations are available in `week6-advanced-development-validation/HealthConnect_Advanced_Analytics_Report.pdf`.

## 🛠️ Tools & Techniques

- Python (pandas, matplotlib, scipy) for statistical validation and advanced analysis
- scikit-learn (LogisticRegression, train/test split, classification metrics) for the cross-track validation model
- Chi-square test of independence for KPI significance validation

## ▶️ Next Steps (Week 7)

- Test the stability of the 4-tier risk segmentation on a separate data subset.
- Evaluate whether adding interaction features (e.g., distance × lead time) improves the validation model.
- Confirm operational feasibility of the risk-based alert system from a Project Management perspective ahead of the final presentation.

## 👤 Author

Josfrid AGBADOGBE, Data Analytics Intern, AnalystLab Africa Experience Lab
*Week 6 — HealthConnect Integration, Advanced Development & Validation*
