import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# STEP 1: GENERATE TRAINING DATA
# We simulate 5000 disaster scenarios
# In Week 6 we replace this with real data
# ─────────────────────────────────────────
print("Generating training data...")
np.random.seed(42)
n = 5000

# Features — what our model will learn from
magnitude        = np.random.uniform(2.5, 9.0, n)
wind_speed       = np.random.uniform(0, 50, n)
temperature      = np.random.uniform(-10, 45, n)
weather_severity = np.random.randint(0, 3, n)   # 0=clear, 1=rain, 2=storm
distress_count   = np.random.randint(0, 10, n)
hour_of_day      = np.random.randint(0, 24, n)  # night disasters are worse
population_proxy = np.random.uniform(0, 1, n)   # 0=remote, 1=dense city

# ─────────────────────────────────────────
# STEP 2: CREATE REALISTIC RISK LABELS
# This is the "ground truth" our model learns from
# ─────────────────────────────────────────
def compute_risk(mag, wind, weather, distress, hour, pop):
    risk = 0
    
    # Magnitude is the strongest signal
    if mag >= 7:   risk += 50
    elif mag >= 6: risk += 35
    elif mag >= 5: risk += 20
    else:          risk += 8
    
    # Wind speed
    if wind > 30:   risk += 20
    elif wind > 15: risk += 10
    
    # Weather severity
    risk += weather * 10
    
    # Human distress signals
    risk += distress * 5
    
    # Night time = harder rescue
    if hour < 6 or hour > 22:
        risk += 10
    
    # Population density
    risk += pop * 15
    
    # Add realistic noise
    risk += np.random.normal(0, 5)
    
    return min(max(risk, 0), 100)

# Generate risk scores
risk_scores = np.array([
    compute_risk(
        magnitude[i], wind_speed[i], weather_severity[i],
        distress_count[i], hour_of_day[i], population_proxy[i]
    )
    for i in range(n)
])

# Convert to binary label: 1 = high risk (>=50), 0 = low risk
labels = (risk_scores >= 50).astype(int)

print(f"High risk scenarios: {labels.sum()} / {n}")
print(f"Low risk scenarios:  {(labels==0).sum()} / {n}")

# ─────────────────────────────────────────
# STEP 3: BUILD DATAFRAME
# ─────────────────────────────────────────
df = pd.DataFrame({
    "magnitude":        magnitude,
    "wind_speed":       wind_speed,
    "temperature":      temperature,
    "weather_severity": weather_severity,
    "distress_count":   distress_count,
    "hour_of_day":      hour_of_day,
    "population_proxy": population_proxy,
    "risk_label":       labels
})

print(f"\nDataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────
# STEP 4: TRAIN/TEST SPLIT
# ─────────────────────────────────────────
FEATURES = [
    "magnitude", "wind_speed", "temperature",
    "weather_severity", "distress_count",
    "hour_of_day", "population_proxy"
]

X = df[FEATURES]
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ─────────────────────────────────────────
# STEP 5: TRAIN XGBOOST MODEL
# ─────────────────────────────────────────
print("\nTraining XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("Training complete!")

# ─────────────────────────────────────────
# STEP 6: EVALUATE MODEL
# ─────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ─────────────────────────────────────────
# STEP 7: SHAP EXPLAINABILITY
# ─────────────────────────────────────────
print("\nGenerating SHAP explanations...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot — shows which features matter most
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_test,
    feature_names=FEATURES,
    show=False
)
plt.title("Feature Importance — What drives disaster risk?")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP summary plot saved as shap_summary.png")

# ─────────────────────────────────────────
# STEP 8: SAVE MODEL
# ─────────────────────────────────────────
with open("disaster_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

print("\nModel saved as disaster_model.pkl")
print("Explainer saved as shap_explainer.pkl")
print("\n✅ Week 4 ML pipeline complete!")
print("Your model can now predict disaster risk AND explain why.")