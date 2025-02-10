import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Models
print('Loading models...')
cnn_model = load_model("Saved/cnn_model.h5") 
autoencoder_classifier = load_model("Saved/autoencoder_classifier.h5")  # Autoencoder classifier
encoder_model = load_model("Saved/encoder.h5")  # Load encoder model to extract latent features

with open("Saved/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load Data
print('Loading dataset...')
df = pd.read_csv('data/cicids_basic_processed.csv')
X = df.drop(columns=['Label', 'Attack Type'])
y = df['Attack Type']

# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = label_encoder.classes_  # Store class names

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_ohe = tf.keras.utils.to_categorical(y_encoded)

# Split Dataset
print('Splitting dataset...')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_ohe, test_size=0.2, random_state=42, shuffle=True)

X_test_enc = X_test.copy()  # Keep the original test set unchanged

# Ensure X_test_enc has the correct shape (73 columns)
while X_test_enc.shape[1] < 73:
    X_test_enc = np.hstack((X_test_enc, X_test_enc[:, : (73 - X_test_enc.shape[1])])) 

# **Extract Latent Features for Autoencoder Classifier**
print('Extracting latent features...')
autoencoder_features_test = encoder_model.predict(X_test_enc)  # Convert input into 16-dimensional latent space

# Get Predictions
print('Predicting...')
cnn_preds = cnn_model.predict(X_test)
xgb_preds = xgb_model.predict_proba(X_test)
autoencoder_preds = autoencoder_classifier.predict(autoencoder_features_test)  # Use latent features for classification

# Weighted Aggregation Based on Accuracy
weights = np.array([0.99, 0.98, 0.95])

print('Calculating final predictions...')
final_probs = (xgb_preds * weights[0]) + (cnn_preds * weights[1]) + (autoencoder_preds * weights[2])
final_preds = np.argmax(final_probs, axis=1)
y_test_true = np.argmax(y_test, axis=1)

# Evaluate Ensemble Model
ensemble_accuracy = accuracy_score(y_test_true, final_preds)
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")

# Compute Class-Wise Accuracy
conf_matrix = confusion_matrix(y_test_true, final_preds)
class_counts = np.sum(conf_matrix, axis=1)
class_correct = np.diagonal(conf_matrix)
class_wise_accuracy = np.nan_to_num(class_correct / class_counts)  # Avoid division errors

# Ensure class_labels and class_wise_accuracy are the same length
if len(class_labels) > len(class_wise_accuracy):
    class_labels = class_labels[:len(class_wise_accuracy)]

print('Generating visualizations...')

# 1. **Plot Class-Wise Accuracy**
plt.figure(figsize=(12, 6))
sns.barplot(x=class_labels, y=class_wise_accuracy * 100, palette="viridis")
plt.xlabel('Class Label')
plt.ylabel('Accuracy (%)')
plt.title('Class-Wise Accuracy')
plt.xticks(rotation=90)
plt.ylim(0, 100)
plt.show()

# 2. **Confusion Matrix Heatmap**
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# 3. **Compare Model Accuracies**
model_names = ['XGBoost', 'CNN', 'Autoencoder', 'Ensemble']
accuracies = [
    accuracy_score(y_test_true, np.argmax(xgb_preds, axis=1)) * 100,
    accuracy_score(y_test_true, np.argmax(cnn_preds, axis=1)) * 100,
    accuracy_score(y_test_true, np.argmax(autoencoder_preds, axis=1)),
    ensemble_accuracy * 100
]

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=model_names, y=accuracies, palette='coolwarm')

# Annotate bars with accuracy values
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                fontsize=12, fontweight='bold', color='black', 
                xytext=(0, 5), textcoords='offset points')  # Add some space above the bars

plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Model Accuracies')
plt.ylim(85, 100)  # Adjust limit for better visualization
plt.show()


# Generate Classification Report
report = classification_report(y_test_true, final_preds, target_names=class_labels)

# Print Report
print("Classification Report:\n", report)

# Save Report to a Text File
with open("classification_report.txt", "w") as f:
    f.write(report)
