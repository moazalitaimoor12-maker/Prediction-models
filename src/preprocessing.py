import pandas as pd
import numpy as np

df = pd.read_csv("input/Car_details_v3.csv") 
print("original shape:", df.shape)

df = df.drop_duplicates()
print("after dropping duplicates:", df.shape)

df = df[(df["fuel"] == "Diesel") | (df["fuel"] == "Petrol")]
df = df[df["owner"] != "Test Drive Car"]
print("after removing CNG/LPG and test drive cars:", df.shape)

df["mileage"] = df["mileage"].str.split(" ").str[0]
df["engine"] = df["engine"].str.split(" ").str[0]
df["max_power"] = df["max_power"].str.split(" ").str[0]

df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
df["engine"] = pd.to_numeric(df["engine"], errors="coerce")
df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")

df["mileage"] = df["mileage"].replace(0, np.nan)
df["max_power"] = df["max_power"].replace(0, np.nan)

df["brand"] = df["name"].str.split(" ").str[0]
df = df.drop(columns=["name", "torque"])

print(df.isnull().sum())
for col in ["mileage", "engine", "max_power", "seats"]:
    df[col] = df[col].fillna(df[col].median())
print("nulls left:", df.isnull().sum().sum())

df = df[df["km_driven"] < 500000]

df.to_csv("input/cleaned_cars.csv", index=False)
print("final shape:", df.shape)
print(df.dtypes)