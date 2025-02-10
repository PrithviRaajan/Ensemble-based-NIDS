import pandas as pd

from imblearn.over_sampling import SMOTE, ADASYN

print('Reading data...')
df = pd.read_csv('data/cicids_basic_processed.csv')

X = df.drop(columns=['Label', 'Attack Type'])
y = df['Attack Type']

print('performing SMOTE...')
smote = SMOTE(sampling_strategy={ 'Brute Force': 20000, 'Web Attack': 10000, 'Bot': 10000 }, random_state=42)

print('performing ADASYN...')
adasyn = ADASYN(sampling_strategy={ 'Infiltration': 5000, 'Heartbleed': 5000 }, random_state=42)

#SMOTE for 'Brute Force', 'Web Attack', and 'Bot'
X_resampled, y_resampled = smote.fit_resample(X, y)
#ADASYN for 'Infiltration' and 'Heartbleed'
X_resampled, y_resampled = adasyn.fit_resample(X_resampled, y_resampled)
print('cancatenating data...')
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
df_resampled.to_csv('data/resampled_data.csv')
print('data ready!')