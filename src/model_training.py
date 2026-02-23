import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import joblib

from data_preprocessing import load_data, preprocess, encode_and_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def categorize_yield(y):
    q1 = y.quantile(0.33)
    q2 = y.quantile(0.66)
    categories = []
    for val in y:
        if val <= q1:
            categories.append('Low')
        elif val <= q2:
            categories.append('Medium')
        else:
            categories.append('High')
    return categories, q1, q2


def train_and_evaluate():
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, encoders, scaler, features = encode_and_split(df)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")

    regression_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=15, random_state=42),
    }

    results = {}
    best_r2 = -999
    best_name = None

    for name, model in regression_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        feat_imp = None
        if hasattr(model, 'feature_importances_'):
            feat_imp = dict(zip(features, [round(float(x), 4) for x in model.feature_importances_]))
        elif hasattr(model, 'coef_'):
            feat_imp = dict(zip(features, [round(float(abs(x)), 4) for x in model.coef_]))

        results[name] = {
            'metrics': {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 4)},
            'feature_importance': feat_imp,
            'type': 'regression'
        }

        print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model

    print("\nTraining Logistic Regression...")

    y_train_cat, q1, q2 = categorize_yield(y_train)
    y_test_cat, _, _ = categorize_yield(y_test)

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train_cat)

    y_pred_cat = log_model.predict(X_test)
    accuracy = accuracy_score(y_test_cat, y_pred_cat)

    results['Logistic Regression'] = {
        'metrics': {'Accuracy': round(accuracy, 4)},
        'feature_importance': None,
        'type': 'classification',
        'thresholds': {'q1': round(float(q1), 2), 'q2': round(float(q2), 2)}
    }

    print(f"  Accuracy={accuracy:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(best_name, os.path.join(MODELS_DIR, 'best_model_name.pkl'))
    joblib.dump(log_model, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    joblib.dump({'q1': q1, 'q2': q2}, os.path.join(MODELS_DIR, 'yield_thresholds.pkl'))

    with open(os.path.join(MODELS_DIR, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest regression model: {best_name} (R2={best_r2:.4f})")
    print("Models saved to", MODELS_DIR)

    return results


if __name__ == '__main__':
    results = train_and_evaluate()
