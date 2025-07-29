import pandas as pd

def add_time_features(df):
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    return df

def transaction_frequency(df):
    user_freq = df.groupby('user_id').size().reset_index(name='transaction_count')
    return df.merge(user_freq, on='user_id', how='left')
