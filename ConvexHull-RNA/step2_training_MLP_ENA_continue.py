import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, matthews_corrcoef
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import pickle

# GPU Selection (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1 if available

# Display options
pd.set_option('display.max_columns', 15)

# Load dataset
df_ = pd.read_csv('ENA/ENA_deduped_NV1368.csv')
print("Original Dataset Shape:", df_.shape)
print(df_['Label'].value_counts())

min_count = df_['Label'].value_counts().min()
print("Minimum count among labels:", min_count)

# Step 3: For each label group, randomly sample min_count records
df = df_.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=min_count, random_state=42))
print(df['Label'].value_counts())
df = shuffle(df, random_state=42)

# Separate features and target
X = df.iloc[:, -1368:]  # Features
y = df['Label']  # Target
print("Feature Shape:", X.shape, "Target Shape:", y.shape)

# Encode the target labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Save the label encoder to disk
with open('best_models/ENA_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("LabelEncoder saved.")

# Train-test split (90% train, 9% validation, 1% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Print dataset shapes
print("Train Set Shape:", X_train.shape, "Train Target Shape:", y_train.shape)
print("Validation Set Shape:", X_val.shape, "Validation Target Shape:", y_val.shape)
print("Test Set Shape:", X_test.shape, "Test Target Shape:", y_test.shape)

# Define the Neural Network model for binary classification
def create_binary_mlp(input_dim, learning_rate):
    model = Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Binary classification loss
        metrics=['accuracy']
    )
    return model

# Model parameters
input_dim = X_train.shape[1]  # Number of features
learning_rate = 0.000001  # Initial learning rate

# Check if the model already exists
model_path = 'best_models/ENA_binary_classification_model.h5'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    # Load the existing model and continue training
    model = load_model(model_path)
    learning_rate = 0.000000001  # Adjust learning rate for continued training
else:
    # Create and compile a new model
    model = create_binary_mlp(input_dim, learning_rate)

# Add EarlyStopping for better convergence
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

# Add ReduceLROnPlateau to dynamically adjust the learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-10,
    verbose=2
)

# Add ModelCheckpoint to save the best model during training
model_checkpoint = ModelCheckpoint(
    filepath='best_models/ENA_binary_classification_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=2
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=2
)

# Load the best model
best_model = load_model(model_path)

# Evaluate the best model
val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)

print(f"Validation Accuracy (Best Model): {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}")
print(f"Test Accuracy (Best Model): {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Predict and evaluate using the best model
y_test_pred = best_model.predict(X_test)
y_test_pred_labels = (y_test_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Also get predictions for validation set for MCC calculation
y_val_pred = best_model.predict(X_val)
y_val_pred_labels = (y_val_pred > 0.5).astype(int)

# Calculate MCC for both validation and test sets
val_mcc = matthews_corrcoef(y_val, y_val_pred_labels)
test_mcc = matthews_corrcoef(y_test, y_test_pred_labels)

print(f"\nMCC Scores:")
print(f"Validation MCC: {val_mcc:.4f}")
print(f"Test MCC: {test_mcc:.4f}")

# Generate a classification report
print("\nTest Set Evaluation (Best Model):")
print(classification_report(y_test, y_test_pred_labels, 
                           target_names=label_encoder.classes_, 
                           digits=4))

# Add MCC to the final summary
print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Validation Set:")
print(f"  - Accuracy: {val_accuracy:.4f}")
print(f"  - Loss: {val_loss:.4f}")
print(f"  - MCC: {val_mcc:.4f}")
print(f"\nTest Set:")
print(f"  - Accuracy: {test_accuracy:.4f}")
print(f"  - Loss: {test_loss:.4f}")
print(f"  - MCC: {test_mcc:.4f}")
print("="*50)

# Optional: Save performance metrics to a file
performance_metrics = {
    'validation_accuracy': val_accuracy,
    'validation_loss': val_loss,
    'validation_mcc': val_mcc,
    'test_accuracy': test_accuracy,
    'test_loss': test_loss,
    'test_mcc': test_mcc
}

# Save metrics to a pickle file
with open('best_models/performance_metrics.pkl', 'wb') as f:
    pickle.dump(performance_metrics, f)
print("\nPerformance metrics saved to 'best_models/performance_metrics.pkl'")

