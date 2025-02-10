import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# 1. Load Data
print('Loading data...')
df = pd.read_csv('data/resampled_data.csv')

# 2. Preprocess Data
X = df.drop(columns=['Label', 'Attack Type'])
y = df['Label']  # Target variable

print('Encoding labels...')
# Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numerical values
y_one_hot = tf.keras.utils.to_categorical(y_encoded)  # Convert to one-hot encoding

print('Scaling features...')
# 3. Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Splitting data...')
# 4. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42, shuffle=True)

# 5. Build Autoencoder Model
input_dim = X_train.shape[1]
latent_dim = 16
num_classes = y_one_hot.shape[1]

print('Building Autoencoder...')
# Encoder
encoder_input = tf.keras.layers.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(64, activation='relu')(encoder_input)
x = tf.keras.layers.Dense(32, activation='relu')(x)
encoder_output = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
encoder = tf.keras.models.Model(encoder_input, encoder_output, name="Encoder")

# Decoder
decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(32, activation='relu')(decoder_input)
x = tf.keras.layers.Dense(64, activation='relu')(x)
decoder_output = tf.keras.layers.Dense(input_dim, activation='sigmoid')(x)  
decoder = tf.keras.models.Model(decoder_input, decoder_output, name="Decoder")

# Autoencoder Model
autoencoder = tf.keras.models.Model(encoder_input, decoder(encoder_output), name="Autoencoder")
autoencoder.compile(optimizer='adam', loss='mse')

# 6. Train Autoencoder
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
print('Training Autoencoder...')
history_ae = autoencoder.fit(
    X_train, X_train,  # Input = Output
    epochs=50,
    batch_size=32,
    validation_data=(X_test, X_test),
    callbacks=[early_stopping],
    verbose=1
)

# Plot Autoencoder Training Loss
plt.plot(history_ae.history['loss'], label='Train Loss')
plt.plot(history_ae.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Autoencoder Training Loss')
plt.show()

# 7. Extract Latent Features for Classification
print('Extracting Features...')
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# 8. Build Classifier Model Using the Encoder
print('Building Classification Model...')
classification_head = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  
])

# 9. Compile and Train Classifier
classification_head.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('Training Classifier...')
history_cls = classification_head.fit(
    X_train_encoded, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_encoded, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Plot Classifier Training Loss
plt.plot(history_cls.history['loss'], label='Train Loss')
plt.plot(history_cls.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Classifier Training Loss')
plt.show()

# 10. Compute Accuracy on Test Set
y_test_pred = np.argmax(classification_head.predict(X_test_encoded), axis=1)
y_test_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_true, y_test_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 11. Compute Class-Wise Accuracy
conf_matrix = confusion_matrix(y_test_true, y_test_pred)
class_counts = np.sum(conf_matrix, axis=1)
class_correct = np.diagonal(conf_matrix)  
class_wise_accuracy = class_correct / class_counts  

# Get class names from LabelEncoder
class_labels = label_encoder.classes_

# Plot Class-Wise Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x=class_labels, y=class_wise_accuracy, palette='viridis')
plt.xlabel('Attack Type')
plt.ylabel('Accuracy')
plt.ylim(0, 1) 
plt.title('Class-Wise Accuracy')
plt.xticks(rotation=45)
plt.show()

# 12. Save Models
print('Saving models...')
encoder.save("encoder.h5")
decoder.save("decoder.h5")
classification_head.save("autoencoder_classifier.h5")
print('Models successfully saved!')
