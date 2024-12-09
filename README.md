Audio Keyword Spotting and Classification Using CNN

Welcome to the Audio Keyword Spotting and Classification Using CNN! This repository contains Python scripts and models for generating an audio dataset, augmenting it with variations, training a Convolutional Neural Network (CNN) for keyword recognition, and testing the trained model.
🔗 Table of Contents

    Overview
    Features
    Requirements
    Installation
    Usage Guide
    Dataset Generation
    Model Training and Testing
    Manual Testing
    Results
    Visualization
    Future Work
    Contributors

📖 Overview

This project demonstrates how to:

    Generate an audio dataset using Text-to-Speech (TTS).
    Apply audio augmentation techniques like noise addition, speed, and pitch adjustments.
    Train a CNN model to recognize keywords or phrases from the audio data.
    Test and deploy the trained model for real-time predictions.

✨ Features

    Dynamic Audio Dataset Creation: Generate keyword-based and sentence-based audio samples with variations.
    Audio Augmentation: Introduce variations in audio (e.g., speed, pitch, noise) for robust training.
    Deep Learning-Based Keyword Spotting: Train a CNN for accurate keyword recognition.
    Manual Testing: Test the model on unseen audio samples with live predictions.

⚙️ Requirements
Libraries and Tools:

    gTTS - For generating audio samples.
    pydub - For audio processing and augmentations.
    librosa - For feature extraction (MFCC).
    tensorflow - For building and training the CNN model.
    numpy, os, pickle - For data processing and utility functions.



🚀 Usage Guide
Step 1: Generate Dataset

    Run the dataset generation script:

    python dataset_generator.py

    The script creates keyword-based audio files in the samples_5 directory and sentence-based files in sample_sen1.

Step 2: Train the CNN Model

    Run the training script:

    python train_model.py

    This script extracts MFCC features, trains the CNN, and saves the model as keyword_spotting_model.h5.

Step 3: Test the Model

    Use the Manual Testing Section in the script for real-time testing:

    python test_model.py

🛠️ Dataset Generation

    Keyword-based Dataset:
        Generates audio files for predefined keywords.
        Applies augmentation (speed, pitch, noise) for diversity.

    Sentence-based Dataset:
        Generates audio files for user-defined sentences.
        Useful for multi-word commands.

📊 Model Training and Testing

    Feature Extraction: Extracts MFCC features to represent audio signals as spectrogram-like structures.

    CNN Architecture:
        Two convolutional layers with ReLU activation and max pooling.
        Dense layers for classification.
        Dropout layers to prevent overfitting.

    Evaluation: Prints model accuracy and saves the model for deployment.

🔍 Manual Testing

Use the predict_keyword function to test predictions on unseen audio:

audio_file_path = '/path/to/test/audio.mp3'
predict_keyword(audio_file_path)

The predicted keyword will be displayed alongside the audio playback.
🎯 Results

    Test Accuracy: The model achieves an accuracy of approximately 90%+ on clean audio samples.
    Augmentation Impact: Data augmentation improves model robustness to variations in pitch, speed, and noise.

📈 Visualization
Model Training Progress

    Plot training and validation accuracy/loss using Matplotlib:

import matplotlib.pyplot as plt

# Example

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

Confusion Matrix

Generate a confusion matrix to visualize model predictions:

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=label_classes)

🚀 Future Work

    Support for multiple languages and accents.
    Integration with a voice assistant system.
    Use transfer learning with pretrained audio models for enhanced accuracy.

🙌 Contributors

    Nishkal Pokar (Data Analyst, Developer)
    Open to contributions! Feel free to submit pull requests or open issues.

#
