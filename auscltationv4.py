import os
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as displaypip
from scipy import signal
from IPython.display import clear_output
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import random
#import jax.numpy as jnp
import pandas as pd
from keras.utils import to_categorical


#seed = 1234

#to packdata
import george_moody_challenge_2022_utility_script as gmc

data_path= r"C:\Users\prone\auscultation\data\training_data"

patient_files = gmc.find_patient_files(data_path)
num_patient_files = len(patient_files)
print("Num patient files: " + str(num_patient_files))

classes = ['Present', 'Unknown', 'Absent']
num_classes = len(classes)

data = []
labels = list()
new_freq = 500

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
    # Load the current patient data and recordings.
    current_patient_data = gmc.load_patient_data(patient_files[i])
    current_recordings, freq = gmc.load_recordings(data_path, current_patient_data, get_frequencies=True)
    
    for j in range(len(current_recordings)):
        resamp_sig = signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * new_freq))
        data.append(resamp_sig)

        # # Create 2 copies of each sound file
        # for _ in range(2):
        #     modified_data.append(resamp_sig)
        #     modified_labels.append(gmc.get_label(gmc.load_patient_data(patient_files[i]))) #labels is out of index here, need to assign label correctly when duplicating 

        #     # Apply modifications to each set of copies
    
        for loss_percentage in range(0, 7, 6):
            modified_signal = simulate_packet_loss(resamp_sig, loss_percentage)
            modified_data.append(modified_signal)
            modified_labels.append(gmc.get_label(gmc.load_patient_data(patient_files[i])))
    # Packet delay simulation
    
        for delay_ms in range(0, 21, 20):
            modified_signal = simulate_packet_delay(resamp_sig, delay_ms)
            modified_data.append(modified_signal)
            modified_labels.append(gmc.get_label(gmc.load_patient_data(patient_files[i])))
    # Sound distortion simulation
    
        for distortion_percentage in range(0, 11, 10):
            modified_signal = simulate_distortion(resamp_sig, distortion_percentage)
            modified_data.append(modified_signal)
            modified_labels.append(gmc.get_label(gmc.load_patient_data(patient_files[i])))

print(modified_labels)

labels = np.array(modified_labels)
data_numpy = np.asarray(modified_data)
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

#print(f"Present = {np.where(np.argmax(labels,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(labels,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(labels,axis=1)==2)[0].shape[0]}")

# Reshape data_padded to 1-dimensional
data_reshaped = data_padded.reshape(data_padded.shape[0], -1)

# Create a DataFrame combining the reshaped data and labels

# print("Data:" + str(data_reshaped[0]))
# print("Data Shape " + str(data_reshaped.shape))
# print("Labels: " + str(labels))
# print("Labels Shape: " + str(labels.shape))

labels = to_categorical(np.argmax(modified_labels, axis=0), num_classes=num_classes)

df = pd.DataFrame()
df['Data'] = pd.Series(data_reshaped.tolist())
#df['Label'] = np.argmax(labels, axis=1)
#df['Label'] = df['Label'].map({0: 'Present', 1: 'Unknown', 2: 'Absent'})
df['Label'] = labels

# Display the DataFrame
print(df)

print(df.head)

df.to_csv('modified_data.csv', index=False)

from sklearn.metrics import f1_score


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

# Define a callback to stop training when learning rate reaches 0.1
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if tf.keras.backend.get_value(self.model.optimizer.lr) <= 0.1:
            self.model.stop_training = True

# Reshape the data_padded array to add an additional dimension
data_reshaped = np.expand_dims(data_padded, axis=2)

# Define a callback to track metrics and display progress
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Calculate and display F1 score
        predictions = self.model.predict(data_reshaped)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        logs['f1_score'] = f1

        # Display progress and metrics
        print(f"Epoch {epoch + 1}/{self.params['epochs']}: "
              f"Loss = {logs['loss']:.4f}, "
              f"Accuracy = {logs['accuracy']:.4f}, "
              f"F1 Score = {f1:.4f}")


# Train the model
history = model.fit(data_reshaped, labels, epochs=100,
                    callbacks=[StopTrainingCallback(), ProgressCallback()])

# Make predictions on the data
predictions = model.predict(data_reshaped)

#add in plot
