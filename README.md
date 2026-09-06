# HealthConnect Clinic — Appointment No-Show Analysis

**AnalystLab Africa Experience Lab | Data Analytics Track — Week 5**
*(Continuity project — builds on Week 4's problem understanding and initial data review)*

## 📌 Project Overview

HealthConnect Clinic is a fictional healthcare provider facing a significant operational challenge: a high rate of missed appointments (no-shows), which leads to inefficient use of appointment slots, repetitive administrative burden, and reduced quality of patient care.

**Central Project Question:** *How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?*

This is a shared, multi-track Experience Lab project. Week 4 focused on problem understanding and planning; **Week 5 moves into practical implementation** — actual data preparation, exploratory analysis, KPI calculation, and evidence-based business insights.

## 🗂️ Repository Structure

```
├── week4-kickoff/
│   ├── Initial_Analysis_Document.pdf         # Dataset overview, data quality, business questions, potential KPIs
│   └── Week4_Project_Summary.pdf
├── week5-analysis/
│   ├── HealthConnect_Appointment_Data_cleaned.csv   # Cleaned dataset (original preserved separately)
│   ├── Initial_HealthConnect_Analytics_Report.pdf  # Full EDA, 4 calculated KPIs, 5 insights, recommendations
│   └── Week5_Project_Summary.pdf
└── README.md
```

## 📊 Dataset

**File:** `HealthConnect_Appointment_Data.csv` (5,000 fictional, anonymized appointment records, 18 variables)
A cleaned copy (`HealthConnect_Appointment_Data_cleaned.csv`) is provided separately; the original file is never modified.

**Cleaning applied:**
- `reminder_channel` missing values (27.32%) recoded to `"No Reminder Sent"` — confirmed structural (coincides exactly with `reminder_sent = "No"`), not a data quality defect.
- `distance_to_clinic_km` and `waiting_time_minutes` marginal missing values (<2%) imputed with the median.

## 🔍 Week 5 — Key Results

| KPI | Result | Signal Strength |
|---|---|---|
| Overall No-Show Rate | 48.46% (51.15% excluding cancellations) | High |
| No-Show Rate by Reminder Channel | SMS 48.0% vs. No Reminder 54.6% | Moderate |
| **No-Show Rate by Booking Lead Time** | **29.5% (0-7 days) → 71.4% (46-60 days)** | **Very Strong** |
| No-Show Rate by Prior No-Show History | 46.3% (0 prior) → 70.3% (3+ prior) | Very Strong |
| No-Show Rate by Distance to Clinic | 48.7% (0-5km) → 70.0% (30-50km) | Strong |

**Headline insight:** Booking lead time is the single strongest predictor of no-shows identified — patients booking 46-60 days in advance are more than twice as likely to miss their appointment (71.4%) as those booking within a week (29.5%). Demographic factors (gender, age, appointment day/time/type) showed no meaningful signal and were deprioritized.

## ❓ Business Questions Addressed (Week 4 + Week 5)

1. What is the overall no-show rate, and how does it vary by patient/appointment characteristics?
2. Does sending a reminder (and its channel) reduce the no-show rate? *(confirmed: yes, modestly, and channel-dependent)*
3. Does booking lead time affect the likelihood of a no-show? *(confirmed: strongest factor identified)*
4. Do patients with a history of no-shows repeat the pattern? *(confirmed: strong, monotonic relationship)*
5. Does distance to the clinic influence attendance? *(confirmed: progressive effect)*
6. Are certain appointment types, days, or time slots more prone to no-shows? *(no meaningful signal found)*

## ✅ Recommendations (Summary)

1. Introduce a mid-point confirmation reminder for appointments booked more than 30 days in advance.
2. Prioritize SMS as the default reminder channel; reassess the value of WhatsApp reminders.
3. Flag patients with 2+ prior no-shows for personalized confirmation calls rather than standard automated reminders.
4. Explore teleconsultation options for patients located more than 20 km from the clinic.
5. Deprioritize demographic segmentation (gender, age, time slot) — no actionable signal detected.

Full details, visualizations, and interpretation are available in `Initial_HealthConnect_Analytics_Report.docx`.

## 🔗 Cross-Track Collaboration

The four validated KPIs (booking lead time, prior no-show history, reminder channel, distance to clinic) were shared with the **Data Science track** to prioritize feature engineering for the no-show prediction model, avoiding effort spent on low-signal demographic variables.

## 🛠️ Tools & Techniques

- Python (pandas, matplotlib) for data cleaning, exploratory analysis, and visualization
- Structured KPI framework: definition → business question → calculation → interpretation

## ▶️ Next Steps (Week 6)

- Build an interactive dashboard consolidating the 4 validated KPIs, with filters by appointment type, distance band, and reminder channel.
- Explore combined-factor interactions (e.g., long lead time **and** prior no-show history) rather than single-factor analysis.

## 👤 Author

Josfrid AGBADOGBE, Data Analytics Intern, AnalystLab Africa Experience Lab
*Week 5 — HealthConnect Solution Development & Implementation*
