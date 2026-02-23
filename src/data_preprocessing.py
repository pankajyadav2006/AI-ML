import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'yield_df.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
