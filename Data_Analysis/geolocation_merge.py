import pandas as pd

def ip_to_int(ip_str):
    try:
        parts = list(map(int, ip_str.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return None

def merge_with_geolocation(fraud_df, ip_df):
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    ip_df = ip_df.dropna()
    ip_df['lower_bound_ip_address'] = pd.to_numeric(ip_df['lower_bound_ip_address'], errors='coerce')
    ip_df['upper_bound_ip_address'] = pd.to_numeric(ip_df['upper_bound_ip_address'], errors='coerce')

    def find_country(ip):
        match = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & (ip_df['upper_bound_ip_address'] >= ip)]
        return match['country'].values[0] if not match.empty else 'Unknown'

    fraud_df['country'] = fraud_df['ip_int'].apply(find_country)
    return fraud_df
