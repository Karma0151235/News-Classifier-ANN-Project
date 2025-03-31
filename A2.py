import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import seaborn as sns

# Step 1: Dataset Loading
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Step 2: Dataset Balancing 
sample_size = 5000
true_df = true_df.sample(n=min(sample_size, len(true_df)), random_state=42)
fake_df = fake_df.sample(n=min(sample_size, len(fake_df)), random_state=42)

# Step 3: Preprocessing: Title and Text Columns Combination
def clean_text(df):
    df['combined_text'] = df['title'] + " " + df['text']
    return df

data_true = clean_text(true_df)
data_fake = clean_text(fake_df)

# Step 4: Label Columns for Classification
data_true['label'] = 1
data_fake['label'] = 0

# Step 5: Dataset Combination and Shuffling 
combined_df = pd.concat([data_true, data_fake], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Step 6: Tokenization and Padding
vocab_size = 5000
max_length = 200
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(combined_df['combined_text'])
sequences = tokenizer.texts_to_sequences(combined_df['combined_text'])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Step 7: Label Preparation 
labels = combined_df['label']

# Step 8: Data Partition (80:20)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Step 9: Neural Network Model Construction
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Step 10: Model Compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# Step 11: Model Training (With Early Stopping)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=2
)

# Step 12: Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Step 13: Classification Report
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 14: Model Performance Visualization
plt.figure(figsize=(14, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Model Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Step 15: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Step 16: Neural Network Architecture Visualization
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
print("Neural network architecture saved as 'model_architecture.png'.")

# Step 17: Example Predictions (Additional)
def predict_news(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0][0]
    return f"{'True News' if prediction > 0.5 else 'Fake News'} (Confidence: {prediction * 100:.2f}%)"

sample_text_fake = (
    "BREAKING: Scientists reveal shocking evidence of a hidden city under the Atlantic Ocean, "
    "claiming it could rewrite human history. According to unnamed sources, the discovery has been "
    "suppressed by governments worldwide. No peer-reviewed studies have confirmed these claims, "
    "but conspiracy theorists are hailing it as the 'find of the century.'"
)

sample_text_true = (
    "In a landmark agreement, leaders from over 100 countries have signed a global pact to protect "
    "the world's oceans. The agreement, finalized during the Global Ocean Summit in 2016, commits "
    "nations to reducing plastic waste, overfishing, and pollution by 2030. Environmental organizations "
    "have praised the move as a significant step toward preserving marine biodiversity."
)

print(f"Fake News Prediction: {predict_news(sample_text_fake)}")
print(f"True News Prediction: {predict_news(sample_text_true)}")