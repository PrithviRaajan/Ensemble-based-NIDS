import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


# # Load and preprocess data
# df = pd.read_csv('data/resampled_data.csv')

# df['Label'] = LabelEncoder().fit_transform(df['Attack Type'])
# #X = df.drop(columns=['Label', 'Attack Type'])

# df.to_csv('data/resampled_data.csv')
# #print(df['Attack Type'].value_counts())
# print('done!')

df = pd.read_csv('data/cicids_basic_processed.csv')
df.info()