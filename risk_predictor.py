import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load trained model and explainer
with open("disaster_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

FEATURES = [
    "magnitude", "wind_speed", "temperature",
    "weather_severity", "distress_count",
    "hour_of_day", "population_proxy"
]

def predict_risk(magnitude, wind_speed, temperature,
                 weather_severity, distress_count,
                 hour_of_day=12, population_proxy=0.5):
    """
    Takes disaster signals and returns:
    - risk_score (0-100)
    - risk_label (HIGH/MEDIUM/LOW)
    - shap explanation (why this score)
    """
    features = np.array([[
        magnitude, wind_speed, temperature,
        weather_severity, distress_count,
        hour_of_day, population_proxy
    ]])

    # Get probability of high risk
    prob = model.predict_proba(features)[0][1]
    risk_score = int(prob * 100)

    # Risk label
    if risk_score >= 70:
        risk_label = "🔴 HIGH"
    elif risk_score >= 40:
        risk_label = "🟠 MEDIUM"
    else:
        risk_label = "🔵 LOW"

    # SHAP explanation for this specific prediction
    shap_vals = explainer.shap_values(features)[0]
    explanation = dict(zip(FEATURES, shap_vals))

    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "probability": round(prob * 100, 1),
        "explanation": explanation
    }

def get_top_factors(explanation, top_n=3):
    """Returns top N factors driving the risk score"""
    sorted_factors = sorted(
        explanation.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    result = []
    for feature, value in sorted_factors:
        direction = "↑ increases" if value > 0 else "↓ decreases"
        result.append(f"{feature}: {direction} risk ({abs(value):.2f})")

    return result

# ─────────────────────────────────────────
# TEST IT
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("RISK PREDICTOR TEST")
    print("="*50)

    scenarios = [
        {
            "name": "Major earthquake in dense city, stormy night",
            "params": dict(magnitude=7.2, wind_speed=35, temperature=15,
                          weather_severity=2, distress_count=8,
                          hour_of_day=2, population_proxy=0.9)
        },
        {
            "name": "Minor tremor in remote area, clear day",
            "params": dict(magnitude=3.1, wind_speed=5, temperature=22,
                          weather_severity=0, distress_count=0,
                          hour_of_day=14, population_proxy=0.1)
        },
        {
            "name": "Medium earthquake, heavy rain, several distress signals",
            "params": dict(magnitude=5.5, wind_speed=20, temperature=10,
                          weather_severity=1, distress_count=4,
                          hour_of_day=8, population_proxy=0.6)
        },
    ]

    for scenario in scenarios:
        print(f"\n📍 {scenario['name']}")
        result = predict_risk(**scenario["params"])
        print(f"   Risk Score: {result['risk_score']}/100")
        print(f"   Risk Level: {result['risk_label']}")
        print(f"   Probability: {result['probability']}%")
        print(f"   Top factors:")
        for factor in get_top_factors(result["explanation"]):
            print(f"   → {factor}")