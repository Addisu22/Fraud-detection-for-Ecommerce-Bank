import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def preprocess_data(df, label_col, scale=True, balance=True):
    df = df.copy()
    df.dropna(inplace=True)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if balance:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)

    return train_test_split(X, y, test_size=0.2, random_state=42)
