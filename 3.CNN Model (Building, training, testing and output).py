#CNN Model (Building, training, testing and output):

import zipfile
import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
from IPython.display import Audio, display

# # Step 1: Unzip the dataset (Assuming your zip file is 'audio_dataset.zip')
# zip_file = '/content/keyword_audio.zip'
# with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#     zip_ref.extractall('audio_files')

# Step 1: Set dataset path
dataset_dir = "/content/audio_files/samples_5"

# Parameters
n_mfcc = 40  # Number of MFCC features
max_pad_len = 40  # Padding size for MFCC
test_size = 0.2  # Split ratio
random_state = 42  # Random seed for reproducibility

# Load dataset and extract MFCC features
def extract_features_and_labels(dataset_dir):
    features = []
    labels = []
    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(subdir, file)
                label = os.path.basename(subdir)
                # Load audio file
                audio, sr = librosa.load(file_path, sr=None)
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
                # Pad/truncate MFCC to fixed length
                if mfcc.shape[1] < max_pad_len:
                    pad_width = max_pad_len - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :max_pad_len]
                features.append(mfcc)
                labels.append(label)
    return np.array(features), np.array(labels)

# Preprocess the dataset
X, y = extract_features_and_labels(dataset_dir)
X = X[..., np.newaxis]  # Add channel dimension for CNN

# Encode the labels
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(n_mfcc, max_pad_len, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save the model and label encoder
model.save('keyword_spotting_model.h5')

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

print("Model and label encoder saved successfully.")

# -------- Manual Testing Section -------- #
# Load the saved model
model = tf.keras.models.load_model('keyword_spotting_model.h5')

# Load the saved label encoder
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def preprocess_audio_for_prediction(audio_path):
    """Preprocess the audio file into a MFCC feature suitable for model prediction."""
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    mfcc = mfcc[..., np.newaxis]  # Add channel dimension
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    return mfcc

def predict_keyword(audio_path):
    """Predict the keyword in the given audio file."""
    # Preprocess the audio
    preprocessed_audio = preprocess_audio_for_prediction(audio_path)
    
    # Make the prediction
    prediction = model.predict(preprocessed_audio)
    predicted_index = np.argmax(prediction)
    predicted_keyword = le.inverse_transform([predicted_index])[0]
    
    # Play the audio for reference
    display(Audio(audio_path))
    
    print(f"Predicted keyword: {predicted_keyword}")

# Example usage
audio_file_path = '/content/sample_sen1/sentence_2.mp3'  # Provide the path to your test audio file
predict_keyword(audio_file_path)

