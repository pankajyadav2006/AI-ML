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
