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
#import jax.numpy as jnp
import librosa
import pandas as pd

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

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        #clear_output(wait=True)
        #plt.plot(self.x, self.losses, label="loss")
        #plt.plot(self.x, self.val_losses, label="val_loss")
        #plt.legend()
        #plt.show();
    def on_train_end(self, epoch, logs={}):
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

plot_losses = PlotLosses()

from tensorflow import keras


def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def build_model(sig_len,n_features, nb_classes, depth=6, use_residual=True):
    input_layer = keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    #model.compile(loss=[macro_double_soft_f1], optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5),tfa.metrics.F1Score(num_classes=3, threshold= 0.5, name='f1_score')])
    return model

def inception_block(prev_layer):
    
    conv1=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv1=tf.keras.layers.BatchNormalization()(conv1)
    conv1=tf.keras.layers.Activation('relu')(conv1)
    #conv1=tf.keras.layers.SpatialDropout1D(rate=0.2)(conv1)
    
    conv3=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv3=tf.keras.layers.BatchNormalization()(conv3)
    conv3=tf.keras.layers.Activation('relu')(conv3)
    conv3=tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, padding = 'same')(conv3)
    conv3=tf.keras.layers.BatchNormalization()(conv3)
    conv3=tf.keras.layers.Activation('relu')(conv3)
    #conv3=tf.keras.layers.SpatialDropout1D(rate=0.2)(conv3)
    
    conv5=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv5=tf.keras.layers.BatchNormalization()(conv5)
    conv5=tf.keras.layers.Activation('relu')(conv5)
    conv5=tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'same')(conv5)
    conv5=tf.keras.layers.BatchNormalization()(conv5)
    conv5=tf.keras.layers.Activation('relu')(conv5)
    #conv5=tf.keras.layers.SpatialDropout1D(rate=0.2)(conv5)
    
    pool= tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
    convmax=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(pool)
    convmax=tf.keras.layers.BatchNormalization()(convmax)
    convmax=tf.keras.layers.Activation('relu')(convmax)
    #convmax=tf.keras.layers.SpatialDropout1D(rate=0.2)(convmax)
    
    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, convmax], axis=1)
    
    return layer_out

def inception_model(sig_len,n_features, output):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features)) 
    
    X = tf.keras.layers.ZeroPadding1D(3)(input_layer)
    
    X = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'same')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(X)
    
    X = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'same')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    
    X = inception_block(X)
    X = inception_block(X)
    
    
    X = tf.keras.layers.MaxPool1D(pool_size=7, strides=2, padding='same')(X)
    
    X = tf.keras.layers.GlobalAveragePooling1D()(X)
    
    
    output_layer = tf.keras.layers.Dense(units=output,activation='softmax')(X)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5),tfa.metrics.F1Score(num_classes=3, threshold= 0.5, name='f1_score')])

    return model


from sklearn.utils.class_weight import compute_class_weight
def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_f1_score', factor=0.1, patience=3, verbose=1, mode='min',
    min_delta=0.0001, cooldown=5, min_lr=0
)

# Validate the model using 5-fold stratified CV (startification based on the ground truth labels)
folds = 5
batch_size = 30
n_epochs = 50
cnt=0
skf = StratifiedKFold(n_splits=folds)
for train_index, val_index in skf.split(data_padded, np.argmax(labels,axis=1)):
    cnt += 1
    print(f"CV fold {cnt}")
    # Split the development data into train and validation
    X_train, X_val = data_padded[train_index], data_padded[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # Use the prevalence of each class to weight the learning of the different samples/recordings -> lower prevalence = higher weight
    new_weights=calculating_class_weights(y_train)
    keys = np.arange(0,y_train.shape[1],1)
    weight_dictionary = dict(zip(keys, new_weights.T[1]))
    
    # Assign the training data to a Tensorflow dataset 
    tf_dataset_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_train,2), np.asarray(y_train)))
    tf_dataset_train = tf_dataset_train.cache()
    tf_dataset_train = tf_dataset_train.batch(batch_size)
    tf_dataset_train = tf_dataset_train.prefetch(tf.data.AUTOTUNE)
    
    # Assign the validation data to a Tensorflow dataset 
    tf_dataset_val = tf.data.Dataset.from_tensor_slices((np.expand_dims(X_val,2), np.asarray(y_val)))
    tf_dataset_val = tf_dataset_val.cache()
    tf_dataset_val = tf_dataset_val.batch(batch_size)
    tf_dataset_val = tf_dataset_val.prefetch(tf.data.AUTOTUNE) 
    
    model = build_model(data_padded.shape[1],1,labels.shape[1])
    #model = inception_model(data_padded.shape[1],1,labels.shape[1])
    model.fit(x=tf_dataset_train, epochs=n_epochs, 
                validation_data=tf_dataset_val,  
                callbacks = [plot_losses], verbose=0,
                class_weight=weight_dictionary
                )
    res = model.evaluate(tf_dataset_val)
    print(f"Training loss = {res[0]}, Validation loss = {res[1]}, \nValidation accuaracy class 1 = {res[2][0]}, \nValidation accuaracy class 2 = {res[2][1]}, \nValidation accuaracy class 3 = {res[2][2]}")


    