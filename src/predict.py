import os
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    model_name = joblib.load(os.path.join(MODELS_DIR, 'best_model_name.pkl'))
    encoders = joblib.load(os.path.join(MODELS_DIR, 'encoders.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    features = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    return model, model_name, encoders, scaler, features


def predict_yield(area, item, year, rainfall, pesticides, avg_temp,
                  model=None, encoders=None, scaler=None, features=None):
    if model is None:
        model, _, encoders, scaler, features = load_model()

    area_enc = encoders['Area'].transform([area])[0]
    item_enc = encoders['Item'].transform([item])[0]

    input_data = pd.DataFrame([[area_enc, item_enc, year, rainfall, pesticides, avg_temp]],
                               columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    feat_imp = None
    if hasattr(model, 'feature_importances_'):
        feat_imp = dict(zip(features, model.feature_importances_))

    return {
        'yield_hg_ha': round(float(prediction), 2),
        'yield_tonnes_ha': round(float(prediction) / 10000, 2),
        'feature_importance': feat_imp
    }


if __name__ == '__main__':
    result = predict_yield('India', 'Rice, paddy', 2020, 1200.0, 50000.0, 25.5)
    print(f"Predicted yield: {result['yield_hg_ha']} hg/ha")
    print(f"Predicted yield: {result['yield_tonnes_ha']} tonnes/ha")
