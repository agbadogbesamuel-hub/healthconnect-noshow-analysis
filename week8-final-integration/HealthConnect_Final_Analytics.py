"""
HealthConnect Final Analytics & Decision Support
AnalystLab Africa Experience Lab — Week 8 (Data Analytics Track)

Purpose
-------
Final consolidation of the Week 5-7 analytics into one reproducible
decision-support run. This script does NOT introduce new analysis themes.
It re-confirms the final KPIs, closes the open items left by Week 7, and
produces every figure used in the final report, dashboard and slides.

What it does
------------
1. FINAL KPI CONFIRMATION  - recompute every KPI shown on the dashboard
                             from the cleaned dataset (no reuse of old code).
2. FIGURE VALIDATION       - re-derive the Week 7 per-segment error figures
                             with both denominators, so each dashboard label
                             states exactly what it measures.
3. CROSS-VALIDATION        - Week 7 open item: confirm the 0.40 threshold
                             with 5-fold stratified CV instead of one split.
4. OPERATIONAL LOAD        - Week 7 open item (for Project Management): how
                             many appointments each threshold flags.
5. FIGURES + JSON          - figures/ (PNG) and outputs/final_kpis.json,
                             which feed the report, the slides and the
                             hand-over to the other tracks.

Input:  ../week5-analysis/HealthConnect_Appointment_Data_cleaned.csv
        (Week 5 cleaned copy — the original dataset is never modified)
Run:    python HealthConnect_Final_Analytics.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Paths are relative to this file (Week 6-7 scripts used a hard-coded
# absolute path that no longer exists in the repository — fixed here).
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "week5-analysis" / "HealthConnect_Appointment_Data_cleaned.csv"
FIG_DIR = HERE / "figures"
OUT_DIR = HERE / "outputs"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

FEATURES = ["booking_lead_days", "previous_no_shows", "distance_to_clinic_km", "reminder_sent_bin"]
THRESHOLDS = {"baseline_0.50": 0.50, "refined_0.40": 0.40}
SEGMENT_ORDER = ["Low Risk", "Medium Risk (history only)", "Medium Risk (lead only)", "High Risk"]

# Chart palette (validated with the dataviz palette validator, light surface)
TEAL = "#00897b"          # refined / primary series
ORANGE = "#d9692a"        # baseline / comparison series
RAMP = ["#5fb8ad", "#2a9d8f", "#1f7a70", "#155a53"]  # ordinal: risk tiers
INK, INK_2, GRID = "#1f2a37", "#52514e", "#e1e0d9"


# ---------------------------------------------------------------------------
# 0. Data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
sub = df[df["appointment_outcome"] != "Cancelled"].copy()
sub["is_noshow"] = (sub["appointment_outcome"] == "No-Show").astype(int)
sub["reminder_sent_bin"] = (sub["reminder_sent"] == "Yes").astype(int)


def risk_segment(frame):
    high_lead = frame["booking_lead_days"] > 30
    history = frame["previous_no_shows"] >= 1
    return np.select(
        [high_lead & history, high_lead & ~history, ~high_lead & history],
        ["High Risk", "Medium Risk (lead only)", "Medium Risk (history only)"],
        "Low Risk",
    )


sub["risk_group"] = risk_segment(sub)


def rate_table(col, bins=None, labels=None, clip=None):
    s = sub[col]
    if clip is not None:
        s = s.clip(upper=clip)
    if bins is not None:
        s = pd.cut(s, bins=bins, labels=labels)
    g = sub.groupby(s, observed=True)["is_noshow"].agg(["mean", "size"])
    return [{"group": str(k), "noshow_rate": round(v["mean"] * 100, 1), "n": int(v["size"])}
            for k, v in g.iterrows()]


# ---------------------------------------------------------------------------
# 1. FINAL KPI CONFIRMATION
# ---------------------------------------------------------------------------
kpis = {
    "dataset": {"appointments": int(len(df)), "expected_attendance": int(len(sub))},
    "outcome_share_pct": (df["appointment_outcome"].value_counts(normalize=True) * 100).round(2).to_dict(),
    "noshow_rate_excl_cancelled_pct": round(sub["is_noshow"].mean() * 100, 2),
    "no_reminder_share_pct": round((df["reminder_sent"] == "No").mean() * 100, 2),
    "lead_time": rate_table("booking_lead_days", [-1, 7, 14, 30, 45, 60],
                            ["0-7 j", "8-14 j", "15-30 j", "31-45 j", "46-60 j"]),
    "history": rate_table("previous_no_shows", [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"]),
    "channel": rate_table("reminder_channel"),
    "distance": rate_table("distance_to_clinic_km", [-0.1, 5, 10, 20, 30, 50],
                           ["0-5 km", "5-10 km", "10-20 km", "20-30 km", "30-50 km"]),
    "risk_segments": [
        {"group": seg,
         "noshow_rate": round(sub.loc[sub.risk_group == seg, "is_noshow"].mean() * 100, 1),
         "n": int((sub.risk_group == seg).sum()),
         "share_pct": round((sub.risk_group == seg).mean() * 100, 1)}
        for seg in SEGMENT_ORDER
    ],
}

print("=" * 72)
print("1. FINAL KPIs (recomputed from the cleaned dataset)")
print("=" * 72)
print(f"No-show rate (all outcomes)        : {kpis['outcome_share_pct']['No-Show']:.2f}%")
print(f"No-show rate (excluding cancelled) : {kpis['noshow_rate_excl_cancelled_pct']:.2f}%")
print(f"Appointments with no reminder      : {kpis['no_reminder_share_pct']:.2f}%")
for block in ["lead_time", "history", "channel", "distance", "risk_segments"]:
    print(f"\n{block}:")
    for row in kpis[block]:
        print(f"  {row['group']:<28} {row['noshow_rate']:>5.1f}%  (n={row['n']})")

# ---------------------------------------------------------------------------
# 2. FIGURE VALIDATION — Week 6/7 split, per-segment errors, two denominators
# ---------------------------------------------------------------------------
X, y = sub[FEATURES], sub["is_noshow"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=42))


model = new_model().fit(X_train, y_train)
proba_test = model.predict_proba(X_test)[:, 1]
test = sub.loc[X_test.index, ["risk_group", "is_noshow"]].copy()


def metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred),
            "flag_rate": float(np.mean(y_pred))}


split_results, segment_errors = {}, {}
for name, t in THRESHOLDS.items():
    pred = (proba_test >= t).astype(int)
    split_results[name] = {k: round(v, 3) for k, v in metrics(y_test, pred).items()}
    rows = []
    for seg in SEGMENT_ORDER:
        m = (test["risk_group"] == seg).values
        actual, p = test["is_noshow"].values[m], pred[m]
        missed = int(((actual == 1) & (p == 0)).sum())
        rows.append({
            "group": seg, "n": int(m.sum()), "true_noshows": int(actual.sum()),
            # Week 7 published metric: missed no-shows / ALL appointments in segment
            "missed_share_of_segment_pct": round(missed / m.sum() * 100, 1),
            # Same errors, divided by TRUE no-shows only (share of no-shows missed)
            "missed_share_of_true_noshows_pct": round(missed / actual.sum() * 100, 1),
            "flag_rate_pct": round(p.mean() * 100, 1),
        })
    segment_errors[name] = rows
split_results["roc_auc"] = round(roc_auc_score(y_test, proba_test), 3)

print("\n" + "=" * 72)
print("2. FIGURE VALIDATION — single split (Week 6/7 reproduction)")
print("=" * 72)
for name in THRESHOLDS:
    print(f"\n{name}: {split_results[name]}")
    for r in segment_errors[name]:
        print(f"  {r['group']:<28} missed/segment={r['missed_share_of_segment_pct']:>5.1f}%  "
              f"missed/true no-shows={r['missed_share_of_true_noshows_pct']:>5.1f}%  "
              f"flagged={r['flag_rate_pct']:>5.1f}%")
print("\nNOTE: the Week 7 '34.3%' (Low Risk, 0.50) is missed no-shows divided by ALL")
print("Low Risk appointments. Divided by true no-shows, the model misses 96.6%")
print("of Low Risk no-shows at 0.50 and still 67.8% at 0.40.")

# ---------------------------------------------------------------------------
# 3. CROSS-VALIDATION of the threshold (Week 7 open item)
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = {name: [] for name in THRESHOLDS}
fold_auc = []
for train_idx, test_idx in skf.split(X, y):
    m = new_model().fit(X.iloc[train_idx], y.iloc[train_idx])
    p = m.predict_proba(X.iloc[test_idx])[:, 1]
    fold_auc.append(roc_auc_score(y.iloc[test_idx], p))
    for name, t in THRESHOLDS.items():
        fold_metrics[name].append(metrics(y.iloc[test_idx], (p >= t).astype(int)))

cv = {}
for name in THRESHOLDS:
    frame = pd.DataFrame(fold_metrics[name])
    cv[name] = {k: {"mean": round(frame[k].mean(), 3), "std": round(frame[k].std(), 3)} for k in frame}
cv["roc_auc"] = {"mean": round(float(np.mean(fold_auc)), 3), "std": round(float(np.std(fold_auc, ddof=1)), 3)}

print("\n" + "=" * 72)
print("3. 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 72)
for name in THRESHOLDS:
    print(f"{name}: " + "  ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}" for k, v in cv[name].items()))
print(f"ROC-AUC: {cv['roc_auc']['mean']:.3f}±{cv['roc_auc']['std']:.3f}")

# ---------------------------------------------------------------------------
# 4. OPERATIONAL LOAD per 1,000 expected appointments (for Project Management)
# ---------------------------------------------------------------------------
def per_1000(name):
    t = THRESHOLDS[name]
    pred = (proba_test >= t).astype(int)
    n = len(pred)
    tp = int(((pred == 1) & (y_test.values == 1)).sum())
    fp = int(((pred == 1) & (y_test.values == 0)).sum())
    return {"flagged": round(pred.sum() / n * 1000), "noshows_caught": round(tp / n * 1000),
            "false_alarms": round(fp / n * 1000)}


high_share = next(r for r in kpis["risk_segments"] if r["group"] == "High Risk")["share_pct"]
load = {name: per_1000(name) for name in THRESHOLDS}
load["high_risk_calls_per_1000"] = round(high_share * 10)

print("\n" + "=" * 72)
print("4. OPERATIONAL LOAD per 1,000 expected appointments")
print("=" * 72)
for name in THRESHOLDS:
    print(f"{name}: {load[name]}")
print(f"High Risk personal calls: {load['high_risk_calls_per_1000']} per 1,000")

# Illustrative scenarios (correlational — to be confirmed by a pilot, not a forecast)
ch = {r["group"]: r for r in kpis["channel"]}
seg = {r["group"]: r for r in kpis["risk_segments"]}
scenarios = {
    "no_reminder_to_sms_rate": round(ch["No Reminder Sent"]["n"] *
                                     (ch["No Reminder Sent"]["noshow_rate"] - ch["SMS"]["noshow_rate"]) / 100),
    "high_risk_to_medium_lead_rate": round(seg["High Risk"]["n"] *
                                           (seg["High Risk"]["noshow_rate"] - seg["Medium Risk (lead only)"]["noshow_rate"]) / 100),
}
print(f"\nScenario A (no-reminder appointments reach the SMS rate): ~{scenarios['no_reminder_to_sms_rate']} fewer no-shows")
print(f"Scenario B (High Risk brought to Medium-lead rate)      : ~{scenarios['high_risk_to_medium_lead_rate']} fewer no-shows")

# ---------------------------------------------------------------------------
# 5. Save JSON
# ---------------------------------------------------------------------------
final = {"kpis": kpis, "model_single_split": split_results, "segment_errors": segment_errors,
         "cross_validation": cv, "operational_load_per_1000": load, "scenarios": scenarios}
(OUT_DIR / "final_kpis.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Calibri", "font.size": 11, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
    "axes.titlelocation": "left",
})


def fr(v, n=1, pct=True):
    """French number format for chart labels (decimal comma)."""
    return f"{v:.{n}f}".replace(".", ",") + ("%" if pct else "")


DISPLAY = {"No Reminder Sent": "Aucun rappel"}
COMMA_AXIS = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))


def bar_rates(ax, rows, title, color=TEAL, colors=None, rotate=False):
    labels = [DISPLAY.get(r["group"], r["group"]) for r in rows]
    vals = [r["noshow_rate"] for r in rows]
    bars = ax.bar(labels, vals, color=colors or color, width=0.62, edgecolor="white", linewidth=2)
    for b, v, r in zip(bars, vals, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, fr(v), ha="center", va="bottom",
                color=INK, fontsize=10.5, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 2, f"n={r['n']:,}".replace(",", " "), ha="center",
                va="bottom", color="white", fontsize=8.5)
    ax.set_ylim(0, 85)
    ax.set_ylabel("Taux de no-show (%)")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if rotate:
        ax.tick_params(axis="x", labelrotation=12)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7.5, 4))
bar_rates(ax, kpis["lead_time"], "Taux de no-show par délai de réservation")
save(fig, "fig1_lead_time.png")

fig, ax = plt.subplots(figsize=(7.5, 4))
bar_rates(ax, kpis["history"], "Taux de no-show par nombre de no-shows précédents")
ax.set_xlabel("No-shows précédents du patient")
save(fig, "fig2_history.png")

channel_rows = sorted(kpis["channel"], key=lambda r: r["noshow_rate"])
fig, ax = plt.subplots(figsize=(7.5, 4))
bar_rates(ax, channel_rows, "Taux de no-show par canal de rappel",
          colors=[TEAL if r["group"] == "SMS" else "#7fc4bb" for r in channel_rows])
save(fig, "fig3_channel.png")

seg_rows = [dict(r, group=r["group"].replace(" (", "\n(")) for r in kpis["risk_segments"]]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bar_rates(ax, seg_rows, "Taux de no-show par segment de risque", colors=RAMP)
save(fig, "fig4_risk_segments.png")

# CV comparison
names = ["Accuracy", "Précision", "Rappel (recall)", "F1-score"]
keys = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(keys))
fig, ax = plt.subplots(figsize=(7.5, 4))
for i, (name, color, label) in enumerate([("baseline_0.50", ORANGE, "Seuil 0,50 (Semaine 6)"),
                                          ("refined_0.40", TEAL, "Seuil 0,40 (raffiné S7, final)")]):
    means = [cv[name][k]["mean"] for k in keys]
    stds = [cv[name][k]["std"] for k in keys]
    bars = ax.bar(x + (i - 0.5) * 0.36, means, 0.34, color=color, label=label,
                  edgecolor="white", linewidth=2, yerr=stds, capsize=3,
                  error_kw={"ecolor": INK_2, "elinewidth": 1})
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, fr(v, 3, False), ha="center",
                fontsize=9.5, color=INK, fontweight="bold")
ax.set_xticks(x, names)
ax.set_ylim(0, 1)
ax.set_title("Validation croisée 5-fold : seuil 0,50 vs 0,40 (moyenne ± écart-type)")
ax.yaxis.set_major_formatter(COMMA_AXIS)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
ax.grid(axis="x", visible=False)
save(fig, "fig5_threshold_cv.png")

# Missed no-shows by segment, corrected denominator
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(SEGMENT_ORDER))
for i, (name, color, label) in enumerate([("baseline_0.50", ORANGE, "Seuil 0,50"),
                                          ("refined_0.40", TEAL, "Seuil 0,40")]):
    vals = [r["missed_share_of_true_noshows_pct"] for r in segment_errors[name]]
    bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=color, label=label,
                  edgecolor="white", linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, fr(v), ha="center",
                fontsize=9.5, color=INK, fontweight="bold")
ax.set_xticks(x, [s.replace(" (", "\n(") for s in SEGMENT_ORDER])
ax.set_ylim(0, 110)
ax.set_ylabel("% des vrais no-shows manqués")
ax.set_title("Part des vrais no-shows manqués par le modèle, par segment")
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
ax.grid(axis="x", visible=False)
save(fig, "fig6_missed_by_segment.png")

# One-page dashboard
fig = plt.figure(figsize=(13.33, 7.5))
fig.patch.set_facecolor("white")
fig.text(0.03, 0.955, "HealthConnect Clinic — Tableau de bord final des no-shows",
         fontsize=19, fontweight="bold", color=INK)
fig.text(0.03, 0.92, "Données : 5 000 rendez-vous (4 737 hors annulations) · KPIs recalculés et validés en Semaine 8",
         fontsize=10.5, color=INK_2)
tiles = [
    (f"{kpis['outcome_share_pct']['No-Show']:.1f}%".replace(".", ","), "Taux de no-show global"),
    ("29,5% → 71,4%", "No-show : délai 0-7 j → 46-60 j"),
    (f"{seg['High Risk']['noshow_rate']:.1f}%".replace(".", ","), f"No-show High Risk ({fr(seg['High Risk']['share_pct'])} des RDV)"),
    (f"{kpis['no_reminder_share_pct']:.1f}%".replace(".", ","), "RDV sans aucun rappel"),
    (f"{cv['refined_0.40']['recall']['mean']:.2f}".replace(".", ","), "Rappel du modèle final (seuil 0,40, CV)"),
]
for i, (value, label) in enumerate(tiles):
    x0 = 0.03 + i * 0.19
    fig.patches.append(matplotlib.patches.FancyBboxPatch(
        (x0, 0.76), 0.175, 0.12, boxstyle="round,pad=0.004,rounding_size=0.01",
        transform=fig.transFigure, facecolor="#eef7f6", edgecolor="none"))
    fig.text(x0 + 0.012, 0.825, value, fontsize=20, fontweight="bold", color="#155a53")
    fig.text(x0 + 0.012, 0.78, label, fontsize=9.5, color=INK_2)

grid = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.70, bottom=0.07, hspace=0.55, wspace=0.28)
ax = fig.add_subplot(grid[0, 0]); bar_rates(ax, kpis["lead_time"], "Délai de réservation")
ax = fig.add_subplot(grid[0, 1]); bar_rates(ax, kpis["history"], "No-shows précédents")
ax = fig.add_subplot(grid[0, 2]); bar_rates(ax, channel_rows, "Canal de rappel",
                                            colors=[TEAL if r["group"] == "SMS" else "#7fc4bb" for r in channel_rows])
ax.tick_params(axis="x", labelsize=8.5)
ax = fig.add_subplot(grid[1, 0]); bar_rates(ax, kpis["distance"], "Distance à la clinique")
ax.tick_params(axis="x", labelsize=8.5)
ax = fig.add_subplot(grid[1, 1])
short = [dict(r, group=g) for r, g in zip(kpis["risk_segments"], ["Low", "Med.\nhist.", "Med.\ndélai", "High"])]
bar_rates(ax, short, "Segment de risque", colors=RAMP)
ax = fig.add_subplot(grid[1, 2])
for i, (name, color, label) in enumerate([("baseline_0.50", ORANGE, "0,50"), ("refined_0.40", TEAL, "0,40")]):
    means = [cv[name][k]["mean"] for k in ["precision", "recall", "f1"]]
    bars = ax.bar(np.arange(3) + (i - 0.5) * 0.36, means, 0.34, color=color, label=f"Seuil {label}",
                  edgecolor="white", linewidth=2)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, fr(v, 2, False), ha="center", fontsize=8.5, color=INK)
ax.set_xticks(np.arange(3), ["Précision", "Rappel", "F1"])
ax.set_ylim(0, 1.05)
ax.set_title("Modèle : seuil 0,50 vs 0,40 (CV 5-fold)")
ax.yaxis.set_major_formatter(COMMA_AXIS)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
ax.grid(axis="x", visible=False)
fig.savefig(FIG_DIR / "dashboard_final.png", dpi=200, facecolor="white")
plt.close(fig)

print(f"\nSaved: {OUT_DIR / 'final_kpis.json'}")
print(f"Saved: {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")
