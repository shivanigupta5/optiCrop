import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Dataset load
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Features aur Label
X = data.drop("label", axis=1)
y = data["label"]

# Model train
model = RandomForestClassifier()
model.fit(X, y)

# Model save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully!")