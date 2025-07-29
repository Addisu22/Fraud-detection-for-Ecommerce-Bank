import pandas as pd

def clean_data(df):
    print("Initial shape:", df.shape)

    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()  # or use fillna for imputation

    # Correct data types
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    df['ip_address'] = df['ip_address'].astype(str)

    return df
