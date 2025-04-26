import joblib
import numpy as np

model = joblib.load("behavior_model.pkl")
le = joblib.load("behavior_label_encoder.pkl")

def classify_behavior(time_spent, response):
    x = np.array([[time_spent, response]])
    pred = model.predict(x)[0]
    return le.inverse_transform([pred])[0]
