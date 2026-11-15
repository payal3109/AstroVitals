"""Personal-baseline anomaly detection using Z-scores and Isolation Forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = [
    "heart_rate_bpm", "sleep_hours", "activity_steps", "mood_score",
    "stress_score", "radiation_index", "bone_health_index", "muscle_health_index"
]
BASELINE_DAYS = 30
ROLLING_WINDOW = 30  # trailing days used as the "recent normal" after the baseline period
MIN_STD = {  # floors stop tiny natural variation from producing huge Z-scores
    "heart_rate_bpm": 1.5, "sleep_hours": 0.25, "activity_steps": 300,
    "mood_score": 0.3, "stress_score": 3.0, "radiation_index": 0.4,
    "bone_health_index": 0.2, "muscle_health_index": 0.2,
}


def _rolling_robust_z(group: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Z-scores against the astronaut's own baseline, adapting to slow drift.

    Days 1-30 use the fixed baseline. Later days are compared with the median
    and MAD of the previous 30 days (current day excluded), so gradual mission
    trends are not flagged but sudden departures still stand out. Median/MAD
    keeps one earlier outlier from inflating the spread.
    """
    z = pd.DataFrame(index=group.index, columns=FEATURES, dtype=float)
    base_med = baseline[FEATURES].median()
    base_std = (1.4826 * (baseline[FEATURES] - base_med).abs().median())
    for pos, idx in enumerate(group.index):
        if group.at[idx, "day"] <= BASELINE_DAYS:
            med, std = base_med, base_std
        else:
            window = group.iloc[max(0, pos - ROLLING_WINDOW):pos][FEATURES]
            med = window.median()
            std = 1.4826 * (window - med).abs().median()
        std = std.fillna(0)
        for f in FEATURES:
            std[f] = max(std[f], MIN_STD[f])
        z.loc[idx] = (group.loc[idx, FEATURES].astype(float) - med) / std
    return z

def analyze_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze each astronaut against their own first 30 days."""
    result = df.copy()
    result["is_anomaly"] = False
    result["anomaly_reasons"] = ""
    result["risk_level"] = "Normal"
    result["recommended_action"] = "Continue routine monitoring."

    for astronaut, group in result.groupby("astronaut_id"):
        group = group.sort_values("day")
        baseline = group[group["day"] <= BASELINE_DAYS]
        if len(baseline) < 10:
            continue

        zscores = _rolling_robust_z(group, baseline)

        # Isolation Forest is fitted only on this astronaut's baseline.
        model = IsolationForest(
            n_estimators=150, contamination=0.05, random_state=42
        )
        model.fit(baseline[FEATURES])
        # Use a stricter baseline-relative score cutoff to reduce false alerts
        # when slow synthetic mission trends shift the overall distribution.
        baseline_scores = model.score_samples(baseline[FEATURES])
        score_cutoff = float(np.quantile(baseline_scores, 0.02))
        iso_flags = model.score_samples(group[FEATURES]) < score_cutoff

        for idx in group.index:
            row = group.loc[idx]
            reasons = []
            z = zscores.loc[idx]

            # Direction-aware, readable flags.
            if z["heart_rate_bpm"] > 4.0:
                reasons.append("heart rate is unusually high versus personal baseline")
            if z["heart_rate_bpm"] < -4.0:
                reasons.append("heart rate is unusually low versus personal baseline")
            if z["sleep_hours"] < -4.0:
                reasons.append("sleep is substantially below personal baseline")
            if z["activity_steps"] < -4.0:
                reasons.append("activity is substantially below personal baseline")
            if z["mood_score"] < -4.0:
                reasons.append("mood score is below personal baseline")
            if z["stress_score"] > 4.0:
                reasons.append("stress score is above personal baseline")
            if z["radiation_index"] > 4.5:
                reasons.append("radiation index is elevated versus simulated baseline")
            # Bone/muscle indices are slow-moving synthetic indicators in this
            # demo; display them, but do not alert on small cumulative drift.

            # Combine interpretable thresholds with multivariate model.
            # Isolation Forest acts as a corroborating signal rather than
            # raising an alert on its own; this avoids excessive alerts from
            # slow synthetic mission trends.
            iso_support = bool(iso_flags[list(group.index).index(idx)])
            flagged = bool(reasons)
            if flagged and iso_support:
                reasons.append("Isolation Forest also identified an unusual multivariate pattern")

            if flagged:
                result.at[idx, "is_anomaly"] = True
                result.at[idx, "anomaly_reasons"] = "; ".join(reasons)
                severity = "High" if len(reasons) >= 2 or any(
                    term in " ".join(reasons) for term in ["heart rate", "radiation"]
                ) else "Moderate"
                result.at[idx, "risk_level"] = severity
                if severity == "High":
                    action = "Recheck measurements and contact the ground medical team promptly."
                else:
                    action = "Review symptoms, consider rest or exercise adjustment, and monitor again."
                result.at[idx, "recommended_action"] = action

    return result
