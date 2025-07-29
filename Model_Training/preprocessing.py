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
    # df = df.copy()
    # df.dropna(inplace=True)

    # X = df.drop(columns=[label_col])
    # y = df[label_col]

    # if scale:
    #     scaler = StandardScaler()
    #     X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # if balance:
    #     sm = SMOTE(random_state=42)
    #     X, y = sm.fit_resample(X, y)

    # return train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        df = df.copy()

        # Drop non-numeric columns or convert datetime columns if needed
        for col in df.select_dtypes(include=['object', 'datetime64']).columns:
            if col != label_col:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col + '_timestamp'] = df[col].astype('int64') // 1e9  # Convert to seconds
                    df.drop(columns=[col], inplace=True)
                except Exception as e:
                    print(f"Could not convert {col}: {e}")
                    df.drop(columns=[col], inplace=True)

        # Drop rows with NaNs (or impute here)
        df.dropna(inplace=True)

        X = df.drop(columns=[label_col])
        y = df[label_col]

        # Scale features
        if scale:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Handle class imbalance
        if balance:
            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)

        # Train-test split
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None, None, None, None

