"""Simple rule-based guidance. This prototype is not a medical device."""

def get_recommendation(row):
    if not row.get("is_anomaly", False):
        return "No unusual pattern detected. Continue routine monitoring."
    reasons = str(row.get("anomaly_reasons", "")).lower()
    if "heart rate" in reasons or "radiation" in reasons or row.get("risk_level") == "High":
        return "Recheck the reading, assess symptoms, and contact the ground medical team promptly."
    if "sleep" in reasons or "stress" in reasons or "mood" in reasons:
        return "Consider a rest period, complete a mood check-in, and monitor the next readings."
    if "activity" in reasons or "muscle" in reasons or "bone" in reasons:
        return "Review the exercise plan with mission health support and monitor recovery."
    return "Review the flagged indicators and seek medical guidance if symptoms are present."
