import numpy as np
import pandas as pd

df = pd.read_csv('data/processed.csv')
print('Data is read.')
col_names = {col: col.strip() for col in df.columns}
df.rename(columns = col_names, inplace = True)
df.drop_duplicates(inplace=True)
print('Duplicates dropped.')
df.replace([np.inf, -np.inf], np.nan, inplace = True)
df = df.where(pd.notna(df), np.nan)
numeric_cols = df.select_dtypes(include = np.number).columns
inf_count = np.isinf(df[numeric_cols]).sum()
df.replace([np.inf, -np.inf], np.nan, inplace = True)
med_flow_bytes = df['Flow Bytes/s'].median()
med_flow_packets = df['Flow Packets/s'].median()

df['Flow Bytes/s'] = df['Flow Bytes/s'].fillna(med_flow_bytes)
df['Flow Packets/s'] = df['Flow Packets/s'].fillna(med_flow_packets)

attack_map = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack � Brute Force': 'Web Attack',
    'Web Attack � XSS': 'Web Attack',
    'Web Attack � Sql Injection': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

df['Attack Type'] = df['Label'].map(attack_map)
print('Attack type column created')
num_unique = df.nunique()
one_variable = num_unique[num_unique == 1]
not_one_variable = num_unique[num_unique > 1].index

dropped_cols = one_variable.index
df = df[not_one_variable]

df.drop(columns=['Unnamed: 0'], inplace=True)
print('Creating csv...')
df.to_csv('data/cicids_basic_processed.csv')
print('File created!')