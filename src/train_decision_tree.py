import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1. load data
df = pd.read_csv("input/cleaned_cars.csv")

# 2. split into inputs (X) and the answer we want to predict (y)
y = df["fuel"]
X = pd.get_dummies(df.drop(columns=["fuel"]), drop_first=True)

# 3. split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. scale the numbers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. decision tree model (same settings as the A1 lab)
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. check how good it is
train_accuracy = model.score(X_train_scaled, y_train)
test_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, test_pred)

print("Training accuracy:", round(train_accuracy * 100, 2), "%")
print("Test accuracy:", round(accuracy * 100, 2), "%")
print("Tested on", len(y_test), "cars the model had never seen.")
print("\nTree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())

print("\nTop 5 feature importances:")
importances = pd.Series(model.feature_importances_, index=X.columns)
for name, score in importances.nlargest(5).items():
    print(f"{name:20s}: {score:.4f}")

print("\nClassification report:")
print(classification_report(y_test, test_pred))

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
    "name": "Decision Tree",
    "type": "classification",
    "model": model,
    "scaler": scaler,
    "columns": list(X.columns),
    "inputs": list(df.drop(columns=["fuel"]).columns),
    "metrics": metrics,
}
joblib.dump(bundle, "models/decision_tree.joblib")
print("\nSaved models/decision_tree.joblib")

# 9. ask the user for a new car
print("\nEnter the details of a new car:")
year = int(input("Year: "))
selling_price = float(input("Selling price in Rs: "))
km_driven = float(input("Km driven: "))
seller_type = input("Seller type (Dealer / Individual / Trustmark Dealer): ")
transmission = input("Transmission (Manual / Automatic): ")
owner = input("Owner (First Owner / Second Owner / Third Owner / Fourth & Above Owner): ")
mileage = float(input("Mileage in kmpl: "))
engine = float(input("Engine in CC: "))
max_power = float(input("Max power in bhp: "))
seats = float(input("Seats: "))
brand = input("Brand (e.g. Maruti, Honda, Hyundai): ")

# 10. put the new car in a table with one row
new_car = pd.DataFrame([{
    "year": year,
    "selling_price": selling_price,
    "km_driven": km_driven,
    "seller_type": seller_type,
    "transmission": transmission,
    "owner": owner,
    "mileage": mileage,
    "engine": engine,
    "max_power": max_power,
    "seats": seats,
    "brand": brand,
}])

# 11. prepare it the same way as the training data
new_car = pd.get_dummies(new_car)
new_car = new_car.reindex(columns=X.columns, fill_value=0)
new_car_scaled = scaler.transform(new_car)

# 12. predict
result = model.predict(new_car_scaled)[0]
print("\nThis car is a", result, "car.")