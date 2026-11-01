import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib

df = pd.read_csv("input/cleaned_cars.csv")
print("shape:", df.shape)

# train on log(price) because prices are skewed
y = np.log(df["selling_price"])
X_raw = df.drop(columns=["selling_price"])
X = pd.get_dummies(X_raw, drop_first=True)
print("X shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

test_pred = model.predict(X_test_scaled)

# convert back to rupees so the results are easy to understand
test_pred_rupees = np.exp(test_pred)
y_test_rupees = np.exp(y_test)

r2 = r2_score(y_test_rupees, test_pred_rupees)
mae = mean_absolute_error(y_test_rupees, test_pred_rupees)
mape = mean_absolute_percentage_error(y_test_rupees, test_pred_rupees)

print("\n========== REGRESSION MODEL SUMMARY ==========")
print(f"Tested on {len(y_test)} cars the model had never seen.")
print(f"The model explains about {r2 * 100:.1f}% of the variation in car prices (R2 = {r2:.3f}).")
print(f"On average, its predicted price is off by about Rs {mae:,.0f}.")
print(f"That is roughly {mape * 100:.1f}% away from the real price.")
print("==============================================")

metrics = {
    "r2": float(r2),
    "r2_log": float(r2_score(y_test, test_pred)),
    "mae": float(mae),
    "mape": float(mape * 100),
    "train_rows": int(len(y_train)),
    "test_rows": int(len(y_test)),
    "features": int(X.shape[1]),
}

# everything this model needs, packed into ONE file
bundle = {
    "name": "Linear Regression",
    "type": "regression",
    "model": model,
    "scaler": scaler,
    "columns": list(X.columns),
    "inputs": list(X_raw.columns),
    "metrics": metrics,
}
joblib.dump(bundle, "models/regression.joblib")
print("Saved models/regression.joblib")