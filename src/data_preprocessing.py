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
