import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

    
print('Loading data...')
# Load Data
df = pd.read_csv('data/cicids_basic_processed.csv')
X = df.drop(columns=['Label', 'Attack Type'])
y = df['Attack Type']

print('Encoding data...')
# Encode Labels
le = LabelEncoder()
y = le.fit_transform(y)

# Standardize Features
sc = StandardScaler()
X = sc.fit_transform(X)

print('Data is being split into train and test set')
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print('Model training...')
# Train XGBoost Model with evaluation set
eval_set = [(X_train, y_train), (X_test, y_test)]
model = xgb.XGBClassifier(objective='multi:softmax', num_class=9, eval_metric='mlogloss', use_label_encoder=False)
history = model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

print('Results are being visualized...')
# Extract training history
results = model.evals_result()

# Plot Training & Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(results['validation_0']['mlogloss'], label='Train Loss')
plt.plot(results['validation_1']['mlogloss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Training & Validation Log Loss')
plt.legend()
plt.grid()
plt.show()

print('Model predicting...')
# Predictions
y_pred = model.predict(X_test)

print('Results are being visualized...')
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(9), yticklabels=range(9))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print('Classwise accuracy calculated...')
# Class-wise Accuracy
classes = np.unique(y_test)
correct_counts = {cls: 0 for cls in classes}
total_counts = {cls: 0 for cls in classes}
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
class_accuracies = {cls: (correct_counts[cls] / total_counts[cls]) * 100 for cls in classes}

# Plot Class-wise Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_accuracies.keys()), y=list(class_accuracies.values()), palette='Blues_r')
plt.xlabel('Class Labels', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Class-wise Prediction Accuracy (%)', fontsize=14)
plt.yticks([])
for i, v in enumerate(class_accuracies.values()):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
plt.show()

print('Classification report...')
# Classification Report
report = classification_report(y_test, y_pred, digits=2)
print("Classification Report:\n", report)

with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as xgboost_model.pkl ✅")
