import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'yield_df.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_data(path=None):
    if path is None:
        path = DATA_PATH
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df


def preprocess(df):
    df = df.dropna()
    q_low = df['hg/ha_yield'].quantile(0.01)
    q_high = df['hg/ha_yield'].quantile(0.99)
    df = df[(df['hg/ha_yield'] >= q_low) & (df['hg/ha_yield'] <= q_high)]
    return df.reset_index(drop=True)


def encode_and_split(df, test_size=0.2):
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    df['Area'] = le_area.fit_transform(df['Area'])
    df['Item'] = le_item.fit_transform(df['Item'])

    encoders = {'Area': le_area, 'Item': le_item}

    features = ['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year',
                'pesticides_tonnes', 'avg_temp']
    X = df[features]
    y = df['hg/ha_yield']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=features, index=X_test.index
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(features, os.path.join(MODELS_DIR, 'feature_names.pkl'))

    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler, features


if __name__ == '__main__':
    df = load_data()
    print("Raw shape:", df.shape)
    df = preprocess(df)
    print("After cleaning:", df.shape)
    X_train, X_test, y_train, y_test, enc, sc, feat = encode_and_split(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Features:", feat)
