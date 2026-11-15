"""Generate clearly simulated astronaut health data for the AstroVitals prototype."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "simulated_health_data.csv"
RNG = np.random.default_rng(42)

def generate_data(n_astronauts=5, days=180):
    rows = []
    for person in range(1, n_astronauts + 1):
        astronaut = f"AST-{person:03d}"
        # Individual normal ranges vary by astronaut.
        hr_base = RNG.uniform(62, 76)
        sleep_base = RNG.uniform(7.0, 8.0)
        activity_base = RNG.uniform(6500, 9500)
        stress_base = RNG.uniform(20, 38)
        mood_base = RNG.uniform(6.8, 8.2)
        bone_base = RNG.uniform(0.0, 0.08)
        muscle_base = RNG.uniform(0.0, 0.10)

        for day in range(1, days + 1):
            # Slow trends represent a synthetic long-mission environment.
            mission_effect = max(0, day - 30) / 150
            heart_rate = RNG.normal(hr_base + 2.5 * mission_effect, 3.5)
            sleep_hours = RNG.normal(sleep_base - 0.25 * mission_effect, 0.55)
            activity_steps = RNG.normal(activity_base * (1 - 0.08 * mission_effect), 850)
            stress = RNG.normal(stress_base + 5 * mission_effect, 7)
            mood = RNG.normal(mood_base - 0.35 * mission_effect, 0.7)
            radiation_index = max(0, RNG.normal(2.0, 0.8))
            bone_health_index = max(0, 100 - (bone_base * day) - RNG.normal(0, 0.3))
            muscle_health_index = max(0, 100 - (muscle_base * day) - RNG.normal(0, 0.4))

            # Inject a few known synthetic events to test detection.
            injected = False
            if day in (67, 68) and person == 2:
                heart_rate += 24
                sleep_hours -= 2.2
                stress += 28
                injected = True
            if day == 112 and person == 4:
                activity_steps *= 0.35
                mood -= 3
                injected = True
            if day == 145 and person == 1:
                radiation_index += 5
                injected = True

            rows.append({
                "astronaut_id": astronaut,
                "day": day,
                "heart_rate_bpm": round(float(np.clip(heart_rate, 40, 180)), 1),
                "sleep_hours": round(float(np.clip(sleep_hours, 0, 14)), 2),
                "activity_steps": int(np.clip(activity_steps, 0, 30000)),
                "mood_score": round(float(np.clip(mood, 1, 10)), 1),
                "stress_score": round(float(np.clip(stress, 0, 100)), 1),
                "radiation_index": round(float(radiation_index), 2),
                "bone_health_index": round(float(bone_health_index), 2),
                "muscle_health_index": round(float(muscle_health_index), 2),
                "injected_test_event": injected,
                "data_type": "SIMULATED"
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} simulated records to {OUT}")
