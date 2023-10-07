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
import librosa
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import george_moody_challenge_2022_utility_script as gmc

data_path= r"C:\Users\prone\auscultation\data\training_data"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)

patient_files = gmc.find_patient_files(data_path)
num_patient_files = len(patient_files)
print("Num patient files: " + str(num_patient_files))

classes = ['Present', 'Unknown', 'Absent']
num_classes = len(classes)

data = []
labels = list()

new_freq = 500

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

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
    #modified_audio = np.clip(modified_audio, -1, 1)
    return modified_audio


    for sample in audio_samples:
            if random.random() < distortion_factor:
                # Apply distortion
                distorted_samples.append(sample * random.uniform(0.5, 1.5))
            else:
                distorted_samples.append(sample)
    return distorted_samples

for i in tqdm.tqdm(range(num_patient_files-100)):
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

        # for loss_percentage in range(0,7,6):
        #     mod_signal = simulate_packet_loss(resamp_sig, loss_percentage)
        #     data.append(mod_signal)
        #     labels.append(current_labels)

        # for delay_ms in range(0,21,20):
        #     mod_signal = simulate_packet_delay(resamp_sig, loss_percentage)
        #     data.append(mod_signal)
        #     labels.append(current_labels)

        # for distortion_percentage in range(0,11,10):
        #     mod_signal = simulate_distortion(resamp_sig, distortion_percentage)
        #     data.append(mod_signal)
        #     labels.append(current_labels)

print(str(len(data)))
print(str(len(labels)))

labels = np.vstack(labels)
data_numpy = np.asarray(data)
print(f"Number of signals = {data_numpy.shape[0]}")



sig_len = []
for i in data:
    sig_len.append(len(i))
    
print("Max signal length: " + str(np.asarray(sig_len).max()))

#0-pad
data_padded = np.zeros((data_numpy.shape[0],np.asarray(sig_len).max()))
for i in tqdm.tqdm(range(data_numpy.shape[0])):
    data_padded [i] = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(data_numpy[i],0),
                                                                    maxlen=np.asarray(sig_len).max(),
                                                                    padding='post',truncating='post', value=0.0)

print(f"Present = {np.where(np.argmax(labels,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(labels,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(labels,axis=1)==2)[0].shape[0]}")

def create_test_data(loss,delay,distortion):
    train_data = []
    train_labels = list()

    for i in tqdm.tqdm(range(num_patient_files-100,num_patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = gmc.load_patient_data(patient_files[i])
        current_recordings, freq = gmc.load_recordings(data_path, current_patient_data, get_frequencies=True)

        for j in range(len(current_recordings)):
            
            resamp_sig = signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * new_freq))

            # Extract labels and use one-hot encoding.
            current_labels = np.zeros(num_classes, dtype=int)
            label = gmc.get_label(current_patient_data)
            if label in classes:
                j = classes.index(label)
                current_labels[j] = 1

            if loss == 0 and delay == 0 and distortion == 0:
                train_data.append(resamp_sig)
                train_labels.append(current_labels)

            if loss != 0:
                for loss_percentage in range(0,loss,10):
                    mod_signal = simulate_packet_loss(resamp_sig, loss_percentage)
                    train_data.append(mod_signal)
                    train_labels.append(current_labels)

            if delay != 0:
                for delay_ms in range(0,delay,33):
                    mod_signal = simulate_packet_delay(resamp_sig, loss_percentage)
                    train_data.append(mod_signal)
                    train_labels.append(current_labels)

            if distortion != 0:
                for distortion_percentage in range(0,distortion,10):
                    mod_signal = simulate_distortion(resamp_sig, distortion_percentage)
                    train_data.append(mod_signal)
                    train_labels.append(current_labels)

    train_labels = np.vstack(train_labels)
    train_data_numpy = np.asarray(train_data)
    print(f"Number of signals = {train_data_numpy.shape[0]}")

    #0-pad 2
    train_data_padded = np.zeros((train_data_numpy.shape[0],np.asarray(sig_len).max()))
    for i in tqdm.tqdm(range(train_data_numpy.shape[0])):
        train_data_padded [i] = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(train_data_numpy[i],0),
                                                                        maxlen=np.asarray(sig_len).max(),
                                                                        padding='post',truncating='post', value=0.0)

    #print(f"Present = {np.where(np.argmax(train_labels,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(train_labels,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(train_labels,axis=1)==2)[0].shape[0]}")

    return train_data_padded, train_labels

df = pd.DataFrame()
df['Data'] = pd.Series(data_padded.tolist())
df['Label'] = np.argmax(labels,axis=1)
df['Label'] = df['Label'].map({0: 'Present', 1: 'Unknown', 2: 'Absent'})

from sklearn.metrics import f1_score

with tf.device('GPU:0'):
    model = tf.keras.Sequential([

        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(data_padded.shape[1],1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(num_classes, activation='softmax')

    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])


    #X_train, X_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=False, random_state=42)
    X_train = data_padded
    y_train = labels


    model.fit(X_train, y_train, epochs=10, batch_size=32)

    model.summary()


    def test(loss, delay, distortion):

        train_data_padded, train_labels = create_test_data(loss,delay,distortion)
        X_test = train_data_padded
        y_test = labels

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels

        classification_report = classification_report(y_test, y_pred_classes)
        print("-----Loss:" + str(loss) + "/ Delay: " + str(delay) + "/ Distortion: " + str(distortion))
        print(classification_report)

    test(0,0,0)

    test(10,0,0)
    test(20,0,0)
    test(30,0,0)

    test(0,10,0)
    test(0,20,0)
    test(0,30,0)

    test(0,0,10)
    test(0,0,20)
    test(0,0,30)


# def visualize_waveform(data,type):
#     plt.plot(data)
#     plt.xlabel('Sample')
#     plt.ylabel('Amplitude')
#     plt.title('Waveform Visualization:' + str(type))
#     plt.show()

# visualize_waveform(data[0], "normal")

# tested_data = data[0]

# packet_loss_audio = simulate_packet_loss(tested_data, 99)

# visualize_waveform(packet_loss_audio, "packet loss")

# packet_delay_audio = simulate_packet_delay(tested_data, 999)

# visualize_waveform(packet_delay_audio, "packet delay")

# distortion_audio = simulate_distortion(tested_data, 99)

# visualize_waveform(distortion_audio , "distortion")


# import sounddevice as sd
# import numpy as np

# def play_sound_recording(data, sample_rate):
#     sd.play(np.array(data), sample_rate)
#     sd.wait()

# play_sound_recording(data[0], 4000)
# play_sound_recording(packet_loss_audio, 4000)
# play_sound_recording(packet_delay_audio, 4000)
# play_sound_recording(distortion_audio, 4000)

# print(data[0])
# print(packet_loss_audio)
# print(packet_delay_audio)
# print(distortion_audio)











