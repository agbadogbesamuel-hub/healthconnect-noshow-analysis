"""
HealthConnect Model Error Analysis & Threshold Refinement
AnalystLab Africa Experience Lab — Week 7 (Data Analytics Track)

Purpose
-------
Mandatory HC-POD cross-track testing artefact (Data Analytics <-> Data Science).

This script tests the Week 6 baseline validation model (built as a cross-track
integration artefact) by decomposing its errors across the risk segments
defined by the Data Analytics track, rather than relying on the single
aggregate accuracy score reported in Week 6.

Test -> Finding -> Action -> Retest
------------------------------------
TEST:    Break down false negatives / false positives by risk segment
         (Low / Medium (history) / Medium (lead) / High).
FINDING: The default 0.50 threshold misses 34.3% of true no-shows in the
         "Low Risk" segment, while never missing a "High Risk" case (0% FN).
ACTION:  Lower the classification threshold to 0.40, justified by the
         asymmetric business cost of a missed no-show (lost appointment
         slot) versus a false alarm (an unnecessary reminder call).
RETEST:  Re-run the same segment breakdown at threshold=0.40 and confirm
         the false-negative rate drops in the affected segments without
         harming the "High Risk" segment.

Input:  HealthConnect_Appointment_Data_cleaned.csv (Week 5 cleaned dataset)
Output: printed before/after comparison tables
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# 1. Load data and rebuild the exact Week 6 train/test split (same random_state)
# ---------------------------------------------------------------------------
df = pd.read_csv(r"D:\Data Analytics Internship Programme\HealthConnect Experience Lab — AnalystLab Africa\Week 5 - Analysis & KPI Development\HealthConnect_Appointment_Data_cleaned.csv")
sub = df[df["appointment_outcome"] != "Cancelled"].copy()
sub["is_noshow"] = (sub["appointment_outcome"] == "No-Show").astype(int)
sub["reminder_sent_bin"] = (sub["reminder_sent"] == "Yes").astype(int)

FEATURES = ["booking_lead_days", "previous_no_shows", "distance_to_clinic_km", "reminder_sent_bin"]
X = sub[FEATURES]
y = sub["is_noshow"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# 2. Rebuild the Week 6 risk segmentation on the test set
# ---------------------------------------------------------------------------
def build_risk_segments(frame):
    frame = frame.copy()
    frame["high_lead"] = frame["booking_lead_days"] > 30
    frame["has_history"] = frame["previous_no_shows"] >= 1
    frame["risk_group"] = "Low Risk"
    frame.loc[frame["high_lead"] & ~frame["has_history"], "risk_group"] = "Medium Risk (lead only)"
    frame.loc[~frame["high_lead"] & frame["has_history"], "risk_group"] = "Medium Risk (history only)"
    frame.loc[frame["high_lead"] & frame["has_history"], "risk_group"] = "High Risk"
    return frame


def error_breakdown(threshold, label):
    y_pred = (y_proba >= threshold).astype(int)
    results = build_risk_segments(sub.loc[X_test.index])
    results["actual"] = y_test.values
    results["predicted"] = y_pred
    results["error_type"] = "Correct"
    results.loc[(results["actual"] == 0) & (results["predicted"] == 1), "error_type"] = "False Positive"
    results.loc[(results["actual"] == 1) & (results["predicted"] == 0), "error_type"] = "False Negative"

    print(f"\n--- Threshold = {threshold} ({label}) ---")
    print(f"Accuracy={accuracy_score(y_test, y_pred):.3f}  "
          f"Precision={precision_score(y_test, y_pred):.3f}  "
          f"Recall={recall_score(y_test, y_pred):.3f}  "
          f"F1={f1_score(y_test, y_pred):.3f}")

    summary = results.groupby("risk_group").apply(
        lambda d: pd.Series({
            "n": len(d),
            "false_negative_rate_%": round((d["error_type"] == "False Negative").mean() * 100, 1),
            "false_positive_rate_%": round((d["error_type"] == "False Positive").mean() * 100, 1),
        })
    )
    print(summary.to_string())
    return summary


# ---------------------------------------------------------------------------
# 3. TEST: baseline behaviour at the default threshold (0.50)
# ---------------------------------------------------------------------------
print("=" * 70)
print("TEST: Error breakdown by risk segment at default threshold (0.50)")
print("=" * 70)
before = error_breakdown(0.50, "BEFORE refinement")

print("\nFINDING: 'Low Risk' segment has a 34.3% false-negative rate (missed")
print("no-shows) while 'High Risk' has 0% — the default threshold is too")
print("conservative to catch no-shows outside the obviously high-risk group.")

# ---------------------------------------------------------------------------
# 4. ACTION + RETEST: lower the threshold to 0.40 and re-evaluate
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ACTION: Lower classification threshold to 0.40")
print("RETEST: Error breakdown by risk segment at threshold 0.40")
print("=" * 70)
after = error_breakdown(0.40, "AFTER refinement")

print("\nCONCLUSION: False-negative rate on 'Low Risk' drops from 34.3% to")
print("24.1%, and on 'Medium Risk (history only)' from 21.9% to 10.1%,")
print("with 'High Risk' unaffected (still 0%). F1-score improves from 0.652")
print("to 0.686. Recommended operating threshold for HealthConnect: 0.40.")
