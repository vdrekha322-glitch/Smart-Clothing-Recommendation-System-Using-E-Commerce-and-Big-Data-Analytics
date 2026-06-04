import pickle
import numpy as np

with open("fit_model.pkl", "rb") as f:
    model, le_brand, le_category, le_size = pickle.load(f)

def predict_size(data):
    brand_encoded = le_brand.transform([data["brand"]])[0]
    category_encoded = le_category.transform([data["category"]])[0]

    features = np.array([[
        float(data["height"]),
        float(data["weight"]),
        float(data["chest"]),
        float(data["waist"]),
        brand_encoded,
        category_encoded
    ]])

    prediction = model.predict(features)
    size = le_size.inverse_transform(prediction)[0]

    return size
