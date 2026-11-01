import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# 1. load data
df = pd.read_csv("input/cleaned_cars.csv")

# 2. split into inputs (X) and the answer we want to predict (y)
y = df["fuel"]
X_raw = df.drop(columns=["fuel"])
X = pd.get_dummies(X_raw, drop_first=True)

# 3. split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. scale the numbers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. train the model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# 6. check how good it is
train_accuracy = model.score(X_train_scaled, y_train)
test_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, test_pred)

print("Training accuracy:", round(train_accuracy * 100, 2), "%")
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Tested on", len(y_test), "cars the model had never seen.")

# 7. save everything in ONE file
cm = confusion_matrix(y_test, test_pred, labels=["Diesel", "Petrol"])
metrics = {
    "accuracy": float(accuracy * 100),
    "correct": int((test_pred == y_test).sum()),
    "train_rows": int(len(y_train)),
    "test_rows": int(len(y_test)),
    "features": int(X.shape[1]),
    "confusion_matrix": cm.tolist(),
}

bundle = {
    "name": "Naive Bayes",
    "type": "classification",
    "model": model,
    "scaler": scaler,
    "columns": list(X.columns),
    "inputs": list(X_raw.columns),
    "metrics": metrics,
}
joblib.dump(bundle, "models/naive_bayes.joblib")
print("Saved models/naive_bayes.joblib")