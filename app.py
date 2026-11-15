import os
import sqlite3
from functools import lru_cache
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from anomaly_detection import analyze_health_data
from recommendations import get_recommendation

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "simulated_health_data.csv"
DB_FILE = ROOT / "data" / "astrovitals.db"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-local-demo-secret")

def db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    DB_FILE.parent.mkdir(exist_ok=True)
    with db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS mood_checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                astronaut_id TEXT NOT NULL,
                day INTEGER NOT NULL,
                mood INTEGER NOT NULL,
                stress INTEGER NOT NULL,
                note TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

@lru_cache(maxsize=1)
def _analyze_cached(mtime):
    raw = pd.read_csv(DATA_FILE)
    return analyze_health_data(raw)

def load_analyzed():
    """Analyze once and reuse; re-runs only if the CSV file changes."""
    if not DATA_FILE.exists():
        from simulate import generate_data
        DATA_FILE.parent.mkdir(exist_ok=True)
        generate_data().to_csv(DATA_FILE, index=False)
    return _analyze_cached(DATA_FILE.stat().st_mtime)

def clamp_int(value, default, low, high):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default

# Create the table at import time so it also works with `flask run` / gunicorn.
init_db()

@app.route("/")
def index():
    df = load_analyzed()
    astronaut_ids = sorted(df["astronaut_id"].unique())
    selected = request.args.get("astronaut", astronaut_ids[0])
    if selected not in astronaut_ids:
        selected = astronaut_ids[0]
    person = df[df["astronaut_id"] == selected].sort_values("day")
    latest = person.iloc[-1].to_dict()
    recent = person.tail(14)
    hr_min, hr_max = recent["heart_rate_bpm"].min(), recent["heart_rate_bpm"].max()
    hr_span = (hr_max - hr_min) or 1.0
    history = []
    for row in recent.to_dict("records"):
        # Bar height (18-70px) scales with the real heart-rate value.
        row["bar_height"] = int(18 + 52 * (row["heart_rate_bpm"] - hr_min) / hr_span)
        history.append(row)
    alerts = person[person["is_anomaly"]].sort_values("day", ascending=False).head(8)
    total_alerts = int(person["is_anomaly"].sum())
    latest_action = get_recommendation(latest)
    return render_template(
        "dashboard.html", astronaut_ids=astronaut_ids, selected=selected,
        latest=latest, alerts=alerts.to_dict("records"),
        total_alerts=total_alerts, latest_action=latest_action,
        history=history
    )

@app.route("/checkin", methods=["POST"])
def checkin():
    valid_ids = set(load_analyzed()["astronaut_id"].unique())
    astronaut_id = request.form.get("astronaut_id", "")
    if astronaut_id not in valid_ids:
        flash("Unknown astronaut profile; check-in not saved.")
        return redirect(url_for("index"))
    day = clamp_int(request.form.get("day"), 180, 1, 180)
    mood = clamp_int(request.form.get("mood"), 5, 1, 10)
    stress = clamp_int(request.form.get("stress"), 5, 1, 10)
    note = request.form.get("note", "")[:300]
    with db() as con:
        con.execute(
            "INSERT INTO mood_checkins (astronaut_id, day, mood, stress, note) VALUES (?, ?, ?, ?, ?)",
            (astronaut_id, day, mood, stress, note)
        )
    flash("Mood check-in saved locally.")
    return redirect(url_for("index", astronaut=astronaut_id))

@app.route("/checkins")
def checkins():
    with db() as con:
        rows = con.execute(
            "SELECT * FROM mood_checkins ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return render_template("checkins.html", checkins=rows)

@app.route("/health")
def health():
    return {"status": "ok", "project": "AstroVitals", "data": "SIMULATED"}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000,
            debug=os.environ.get("FLASK_DEBUG") == "1")
