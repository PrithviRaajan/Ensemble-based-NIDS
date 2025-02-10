import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Load and preprocess data
df = pd.read_csv('data/cicids_basic_processed.csv')
X = df.drop(columns=['Label', 'Attack Type','Unnamed: 0'])

print('Dataframe prepared.')

y = LabelEncoder().fit_transform(df['Attack Type'])
X = StandardScaler().fit_transform(X)
y_ohe = tf.keras.utils.to_categorical(y)

print('Encoding and Scaling done.')
X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.2, random_state=42, shuffle=True)
print('Split into train and test sets')
# Define CNN model
model = models.Sequential([
    layers.InputLayer(input_shape=(70, 1)),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Dropout(0.5),
    layers.Conv1D(64, 3, activation='relu'),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(9, activation='softmax')
])
print('Model architecture established.')
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print('Model is now compiled')
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Extract accuracy and validation accuracy from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)
print('plotting chart...')
# Plot the training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()
plt.show()

# Predictions
print("Predicting...")
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print('Plotting confusion matrix..')
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_labels, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=range(9), yticklabels=range(9))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print('Accuracies are being charted...')
# Class-wise Accuracy
classes = np.unique(y_test_labels)
correct_counts = {cls: 0 for cls in classes}
total_counts = {cls: 0 for cls in classes}
for true, pred in zip(y_test_labels, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
class_accuracies = {cls: (correct_counts[cls] / total_counts[cls]) * 100 for cls in classes}

# Bar Chart for Class-wise Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_accuracies.keys()), y=list(class_accuracies.values()), palette="Blues_r")
plt.xlabel("Class Labels", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Class-wise Prediction Accuracy (%)", fontsize=14)
plt.ylim(0, 120)
plt.yticks([])
for i, v in enumerate(class_accuracies.values()):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test_labels, y_pred, digits=2))

print('Saving model...')
model.save('saved_models/cnn.h5')

print('CNN Model saved successfully!')