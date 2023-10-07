import os
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as displaypip
import tensorflow_addons as tfa
from scipy import signal
from IPython.display import clear_output
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import random
import librosa
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import george_moody_challenge_2022_utility_script as gmc
data_path = r"C:\Users\prone\auscultation\data\training_data"
# Define the functions `find_patient_files`, `load_patient_data`, and `load_recordings` as required.
patient_files = gmc.find_patient_files(data_path)
num_patient_files = len(patient_files)
print("Num patient files: " + str(num_patient_files))

classes = ['Present', 'Unknown', 'Absent']
num_classes = len(classes)

data = []
labels = list()
new_freq = 500


for i in tqdm.tqdm(range(num_patient_files)):
    # Load the current patient data and recordings.
    current_patient_data = gmc.load_patient_data(patient_files[i])
    current_recordings, freq = gmc.load_recordings(data_path, current_patient_data, get_frequencies=True)
    
    for j in range(len(current_recordings)):
        
        resamp_sig = signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * new_freq))
        data.append(resamp_sig)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = gmc.get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
        labels.append(current_labels)

labels = np.vstack(labels)
data_numpy = np.asarray(data)
print(f"Number of signals = {data_numpy.shape[0]}")

sig_len = []
for i in tqdm.tqdm(data):
    sig_len.append(len(i))
    
print("Max signal length: " + str(np.asarray(sig_len).max()))

#0-pad
data_padded = np.zeros((data_numpy.shape[0],np.asarray(sig_len).max()))
for i in tqdm.tqdm(range(data_numpy.shape[0])):
    data_padded [i] = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(data_numpy[i],0),
                                                                    maxlen=np.asarray(sig_len).max(),
                                                                    padding='post',truncating='post', value=0.0)

print(f"Present = {np.where(np.argmax(labels,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(labels,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(labels,axis=1)==2)[0].shape[0]}")

---------
# Function to create modified versions of audio signals
def create_modified_signals(audio, label):
    modified_signals = []
    if label == 'packet loss':
        for loss_percentage in range(0, 51, 10):
            modified_signal = simulate_packet_loss(audio, loss_percentage)
            modified_signals.append(modified_signal)
    elif label == 'packet delay':
        for delay_ms in range(0, 1001, 100):
            modified_signal = simulate_packet_delay(audio, delay_ms)
            modified_signals.append(modified_signal)
    elif label == 'distortion':
        for distortion_percentage in range(0, 31, 6):
            modified_signal = simulate_distortion(audio, distortion_percentage)
            modified_signals.append(modified_signal)
    return modified_signals

# Function to simulate packet loss on audio signal
def simulate_packet_loss(audio, loss_percentage):
    audio_length = len(audio)
    num_samples_to_remove = int(audio_length * loss_percentage / 100)
    samples_to_remove = random.sample(range(audio_length), num_samples_to_remove)
    modified_audio = np.delete(audio, samples_to_remove)
    return modified_audio

# Function to simulate packet delay on audio signal
def simulate_packet_delay(audio, delay_ms):
    delay_samples = int(delay_ms * new_freq / 1000)
    modified_audio = np.pad(audio, (delay_samples, 0))
    modified_audio = modified_audio[:len(audio)]
    return modified_audio

# Function to simulate distortion on audio signal
def simulate_distortion(audio, distortion_percentage):
    modified_audio = audio + (distortion_percentage / 100) * np.random.normal(0, 1, len(audio))
    modified_audio = np.clip(modified_audio, -1, 1)
    return modified_audio

# Generate modified versions of audio signals
modified_data = []
modified_labels = []
for i in tqdm.tqdm(range(num_patient_files)):
    # Load the current patient data and recordings
    current_patient_data = gmc.load_patient_data(patient_files[i])
    current_recordings, freq = gmc.load_recordings(data_path, current_patient_data, get_frequencies=True)

    for j in range(len(current_recordings)):
        resamp_sig = signal.resample(current_recordings[j], int((len(current_recordings[j]) / freq[j]) * new_freq))

        # Create 5 versions of the same audio signal
        for k in range(5):#dont need labels
            modified_data.append(resamp_sig)
            if k == 0:
                modified_labels.append('packet loss')
            elif k == 1:
                modified_labels.append('packet delay')
            elif k == 2:
                modified_labels.append('distortion')
            else:
                modified_labels.append('original')

# Apply modifications to the audio signals based on the labels
final_data = []
final_labels = []
for audio, label in zip(modified_data, modified_labels):
    modified_signals = create_modified_signals(audio, label)
    for modified_signal in modified_signals:
        final_data.append(modified_signal)
        final_labels.append(label)

# Convert the final_data and final_labels to NumPy arrays
# final_data = np.asarray(final_data)
# final_labels = np.asarray(final_labels)
# Convert the final_data and final_labels to NumPy arrays
final_data = [np.asarray(signal) for signal in final_data]
final_data = np.asarray(final_data)
final_labels = np.asarray(final_labels)


# Shuffle the data and labels in unison
random_state = np.random.get_state()
np.random.shuffle(final_data)
np.random.set_state(random_state)
np.random.shuffle(final_labels)

print(final_data)
print(final_data.size)
print(final_labels)

# Rest of code
# ...
-----------
# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(data_padded.shape[1], 1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Define a callback to stop training when the learning rate reaches 0.1
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if tf.keras.backend.get_value(self.model.optimizer.lr) <= 0.1:
            self.model.stop_training = True

# Reshape the data_padded array to add an additional dimension
print(final_data.shape)
data_reshaped = np.expand_dims(final_data, axis=-1)

# Define a callback to track metrics and display progress
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Calculate and display F1 score
        predictions = self.model.predict(data_reshaped)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(final_labels, axis=1)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        logs['f1_score'] = f1

        # Display progress and metrics
        print(f"Epoch {epoch + 1}/{self.params['epochs']}: "
              f"Loss = {logs['loss']:.4f}, "
              f"Accuracy = {logs['accuracy']:.4f}, "
              f"F1 Score = {f1:.4f}")

        # Plot accuracy, loss, and F1 score
        history = self.model.history.history
        epochs = range(1, epoch + 2)

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 12))
        ax1.plot(epochs, history['accuracy'], label='Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(epochs, history['loss'], label='Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        ax3.plot(epochs, history['f1_score'], label='F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()

        plt.tight_layout()
        plt.show()

# Train the model
history = model.fit(data_reshaped, final_labels, epochs=100,
                    callbacks=[StopTrainingCallback(), ProgressCallback()])

# Make predictions on the data
predictions = model.predict(data_reshaped)


# Convert predicted labels from one-hot encoding to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(final_labels, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

predicted_probabilities = model.predict(data_reshaped)
max_probabilities = np.max(predicted_probabilities, axis=1)
plt.figure(figsize=(8, 6))
plt.hist(max_probabilities, bins=10, range=(0, 1))
plt.xlabel('Maximum Probability')
plt.ylabel('Frequency')
plt.title('Confidence Histogram')
plt.show()

