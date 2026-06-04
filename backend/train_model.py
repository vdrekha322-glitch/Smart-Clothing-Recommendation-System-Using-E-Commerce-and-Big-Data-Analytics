import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
data_path = os.path.join("data", "clothing_fit_data.csv")
data = pd.read_csv(data_path)

# Encode categorical columns
le_brand = LabelEncoder()
le_category = LabelEncoder()
le_size = LabelEncoder()

data["brand"] = le_brand.fit_transform(data["brand"])
data["category"] = le_category.fit_transform(data["category"])
data["size"] = le_size.fit_transform(data["size"])

X = data.drop("size", axis=1)
y = data["size"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
with open("fit_model.pkl", "wb") as f:
    pickle.dump((model, le_brand, le_category, le_size), f)

print("✅ Model trained and saved as fit_model.pkl")
