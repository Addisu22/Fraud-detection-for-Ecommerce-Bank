from Data_Analysis.config import fraud_data_path, ip_data_path
from Data_Analysis.utils import load_csv

def load_datasets():
    fraud_df = load_csv(fraud_data_path)
    ip_df = load_csv(ip_data_path)
    return fraud_df, ip_df
