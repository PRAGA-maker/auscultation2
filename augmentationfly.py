import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from scipy import signal
from sklearn.metrics import classification_report
import george_moody_challenge_2022_utility_script as gmc
from sklearn.metrics import accuracy_score, f1_score, recall_score


def simulate_packet_loss(audio, loss_percentage):
    audio_length = len(audio)
    num_samples_to_remove = int(audio_length * loss_percentage / 100)
    samples_to_remove = random.sample(range(audio_length), num_samples_to_remove)
    modified_audio = np.delete(audio, samples_to_remove)
    return modified_audio


def simulate_packet_delay(audio, delay_ms):
    delay_samples = int(delay_ms * new_freq / 1000)
    modified_audio = np.pad(audio, (delay_samples, 0))
    modified_audio = modified_audio[:len(audio)]
    return modified_audio


def simulate_distortion(audio, distortion_percentage):
    modified_audio = audio + (distortion_percentage / 100) * np.random.normal(0, 1, len(audio))
    return modified_audio


def data_generator(batch_size, data, labels):
    while True:
        idx = np.random.choice(len(data), batch_size)
        
        batch_data = []
        batch_labels = []
        
        for i in idx:
            audio = data[i]
            label = labels[i]
            
            aug_type = np.random.choice(['loss', 'delay', 'distortion', 'none'])
            
            if aug_type == 'loss':
                loss_percentage = np.random.choice([10, 20, 30])
                audio = simulate_packet_loss(audio, loss_percentage)
                
            elif aug_type == 'delay':
                delay_ms = np.random.choice([33, 66, 99])
                audio = simulate_packet_delay(audio, delay_ms)
                
            elif aug_type == 'distortion':
                distortion_percentage = np.random.choice([10, 20, 30])
                audio = simulate_distortion(audio, distortion_percentage)
            
            batch_data.append(audio)
            batch_labels.append(label)
            
        batch_data = np.asarray(batch_data)
        batch_labels = np.asarray(batch_labels)
        
        yield batch_data, batch_labels


# ...[Some parts of your original script remain unchanged]...

# Set constants and data paths
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
data_path= r"C:\Users\praneelpatel\Documents\GitHub\PRAGA\mod_data\training_data"
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

    phone_low_freq = 20 #20Hz
    phone_high_freq = 20000 #20kHz

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

    # for sample in audio_samples:
    #         if random.random() < distortion_factor:
    #             # Apply distortion
    #             distorted_samples.append(sample * random.uniform(0.5, 1.5))
    #         else:
    #             distorted_samples.append(sample)
    # return distorted_samples

for i in range(num_patient_files-100): #USED TO BE 100, tqdm
    # Load the current patient data and recordings.
    current_patient_data = gmc.load_patient_data(patient_files[i])
    current_recordings, freq = gmc.load_recordings(data_path, current_patient_data, get_frequencies=True)

    #print(current_patient_data)
    #print((str(len(current_recordings))+ str(" -- len current recordings")))
    #print(str((len(freq)))+ str(" -- freq"))

    for j in range(len(current_recordings)):
        
        resamp_sig = signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * new_freq))

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = gmc.get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
        labels.append(current_labels)
        
        #equalization, filtering
        #ambient sounds
        #"Compression and Dynamics: meaning loud sounds and soft sounds are closer in volume. Apply gentle compression to the recording 
        compression_factor = random.randint(0,1000)
        compressed_sig = np.sign(resamp_sig) * np.log1p(compression_factor*np.abs(resamp_sig))
        #visualize_waveform(compressed_sig, "compression")

        #"Distortion and Saturation: mimic the slightly distorted and saturated sound of some phone recordings,add distortion or saturation to the recording.
        distorted_sig = np.clip(compressed_sig, -20000, 20000)
        noise = np.random.normal(0,200,len(distorted_sig))
        noised_sig = distorted_sig + noise
        #visualize_waveform(noised_sig, "noised_sig")

        # reduce frequencies at 500Hz and below by 15dB, reduce frequencies at 6,000Hz and above by 15dB, boost frequencies from 600Hz to 5,000Hz by 2dB (from https://voiceovertip.com/how-to-make-voice-recording-sound-like-a-phone-call/)
        order = 4 

        try:
            sample_rate = int((len(current_recordings[j])/freq[j]) * new_freq)
            print(sample_rate)
        except IndexError:
            print("ERROR: " + str(j))
            #print(current_recordings[j]) #<-- issue boy
            #print(freq[j])
        
        #print("SR: " + str(sample_rate))
        b_low, a_low = signal.butter(order, 500 / (sample_rate / 2), btype='low', analog=False)
        #print("B_Low: " + str(b_low))
        #print("A_Low: " + str(a_low))
        if sample_rate <= 12000:
            sample_rate = 12001
            #print("New SR: " + str(sample_rate))
        b_high, a_high = signal.butter(order, 6000 / (sample_rate / 2), btype='high', analog=False)
        #print("B_high: " + str(b_high))
        #print("A_high: " + str(a_high))
        target_sample_rate = 6000 / (sample_rate /2) in [0,1]
        #print("TSR: " + str(target_sample_rate))
        b_band, a_band = signal.butter(order, [600 / (sample_rate / 2), 5000 / (sample_rate / 2)], btype='band', analog=False)        
        filtered_low = signal.lfilter(b_low, a_low, noised_sig)
        filtered_high = signal.lfilter(b_high, a_high, noised_sig)
        filtered_band = signal.lfilter(b_band, a_band, noised_sig)
        gain_low = 10 ** (-15 / 20)  # -15dB to gain factor
        gain_high = 10 ** (-15 / 20)  # -15dB to gain factor
        gain_band = 10 ** (2 / 20)    # 2dB to gain factor
        adjusted_low = filtered_low * gain_low
        adjusted_high = filtered_high * gain_high
        adjusted_band = filtered_band * gain_band
        freqd_sig = adjusted_band + adjusted_high + adjusted_low

        freqd_sig = resamp_sig

        data.append(resamp_sig)

        for loss_percentage in range(0,31,10):#change resamp sig to end result
            mod_signal = simulate_packet_loss(resamp_sig, loss_percentage)
            data.append(mod_signal)
            labels.append(current_labels)

        for delay_ms in range(0,100,33):
            mod_signal = simulate_packet_delay(resamp_sig, delay_ms)
            data.append(mod_signal)
            labels.append(current_labels)

        for distortion_percentage in range(0,31,10):
            mod_signal = simulate_distortion(resamp_sig, distortion_percentage)
            data.append(mod_signal)
            labels.append(current_labels)

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

df = pd.DataFrame({'sound': data, 'label': np.argmax(labels, axis=1)})

class_names = ['Present', 'Absent', 'Unknown']
df['label'] = df['label'].map(lambda x: class_names[x])

X_train = data_padded
y_train = labels

# Create a generator instance
batch_size = 32
train_gen = data_generator(batch_size, X_train, y_train)

steps_per_epoch = len(X_train) // batch_size

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(data_padded.shape[1],1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model using the generator
model.fit(train_gen, epochs=1, steps_per_epoch=steps_per_epoch)

model.summary()

def create_test_data(loss,delay,distortion):
    train_data = []
    train_labels = list()
    train_data_padded = []

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
                mod_signal = simulate_packet_loss(resamp_sig, loss)
                train_data.append(mod_signal)
                train_labels.append(current_labels)

            if delay != 0:
                mod_signal = simulate_packet_delay(resamp_sig, delay)
                train_data.append(mod_signal)
                train_labels.append(current_labels)

            if distortion != 0:
                mod_signal = simulate_distortion(resamp_sig, distortion)
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

def test(loss, delay, distortion):

    train_data_padded, train_labels = create_test_data(loss,delay,distortion)
    X_test = train_data_padded
    y_test = train_labels

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels

    y_test_binary = np.argmax(y_test, axis=1)

    print("-----Loss:" + str(loss) + "/ Delay: " + str(delay) + "/ Distortion: " + str(distortion))
    print(classification_report(y_test_binary, y_pred_classes))

    accuracy = accuracy_score(y_test_binary,y_pred_classes)
    print("Accuracy:", accuracy)

    f1 = f1_score(y_test_binary, y_pred_classes, average='weighted')
    print("F1-score:", f1)

    sensitivity_0 = recall_score(y_test_binary, y_pred_classes, average=None)[0]
    sensitivity_1 = recall_score(y_test_binary, y_pred_classes, average=None)[1]
    sensitivity_2 = recall_score(y_test_binary, y_pred_classes, average=None)[2]

    print("Sensitivitty: " + str((sensitivity_0+sensitivity_1+sensitivity_2)/3))

test(0,0,0)

test(10,0,0)
test(20,0,0)
test(30,0,0)

test(0,33,0)
test(0,66,0)
test(0,99,0)

test(0,0,10)
test(0,0,20)
test(0,0,30)

test(30,99,30)