from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# allow the HTML page to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load one file per model, once, when the server starts
model_files = [
    "regression",
    "decision_tree",
    "logistic_regression",
    "svm",
    "naive_bayes",
    "random_forest",
]

bundles = {}
for key in model_files:
        bundles[key] = joblib.load("models/" + key + ".joblib")


# the inputs each endpoint expects
class RegressionInput(BaseModel):
    year: int
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float
    brand: str


class ClassificationInput(BaseModel):
    algorithm: str          # which classification model to use
    year: int
    selling_price: float
    km_driven: float
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float
    brand: str


# same preparation steps for every model
def prepare(row_dict, bundle):
    row = pd.DataFrame([row_dict])
    row = pd.get_dummies(row)
    row = row.reindex(columns=bundle["columns"], fill_value=0)
    return bundle["scaler"].transform(row)


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    bundle = bundles["regression"]
    X = prepare(data.model_dump(), bundle)
    # model predicts log price, so convert back to rupees
    price = float(np.exp(bundle["model"].predict(X)[0]))
    return {"predicted_price": round(price)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    if data.algorithm not in bundles or bundles[data.algorithm]["type"] != "classification":
        raise HTTPException(status_code=400, detail="Unknown classification model")

    bundle = bundles[data.algorithm]
    row = data.model_dump()
    row.pop("algorithm")        # not a car feature, so remove it
    X = prepare(row, bundle)
    fuel = str(bundle["model"].predict(X)[0])
    return {"algorithm": bundle["name"], "predicted_fuel": fuel}


# the page uses this to build the menu, stats and disabled inputs
@app.get("/models")
def get_models():
    result = {}
    for key, bundle in bundles.items():
        result[key] = {
            "name": bundle["name"],
            "type": bundle["type"],
            "inputs": bundle["inputs"],
            "metrics": bundle["metrics"],
        }
    return result