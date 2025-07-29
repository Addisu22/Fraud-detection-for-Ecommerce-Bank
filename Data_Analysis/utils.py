import pandas as pd
import numpy as np
from datetime import datetime

def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found at {path}")
        return pd.DataFrame()