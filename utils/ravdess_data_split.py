import numpy as np
import pandas as pd
import math
import sys
import os
import glob
from random import shuffle
import h5py
import librosa
import pickle
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from utilities import read_audio, split_actor
from statistics import mean, median, stdev
import config

hdf5_path = os.path.join('/storage/home/ychang/RAVDESS', 'ravdess.h5')
x_train = []
y_train = []
intensity_train = []
x_val = []
y_val = []
intensity_val = []
x_test = []
y_test = []
intensity_test = []

actors = list(range(1, 25))
train_actor, val_actor, test_actor = split_actor(actors)
print('{} actors for train:'.format(len(train_actor)))
print(train_actor)
print('{} actors for validation:'.format(len(val_actor)))
print(val_actor)
print('{} actors for test:'.format(len(test_actor)))
print(test_actor)

# path to data for glob
data_path = '/storage/home/ychang/RAVDESS/Actor_*/*.wav'
# RAVDESS dataset emotions
# shift emotions left to be 0 indexed for PyTorch
emotions_dict ={
    '0':'surprised',
    '1':'neutral',
    '2':'calm',
    '3':'happy',
    '4':'sad',
    '5':'angry',
    '6':'fearful',
    '7':'disgust'
}

# Additional attributes from RAVDESS to play with
emotion_attributes = {
    '01': 'normal',
    '02': 'strong'
}

dur = []
file_count = 0
for file in glob.glob(data_path):
    # get file name with labels
    file_name = os.path.basename(file)

    # get the actor name
    actor = int(file_name.split("-")[-1].split(".")[0])

    # get emotion label from the sample's file
    emotion = int(file_name.split("-")[2])

    #  move surprise to 0 for cleaner behaviour with PyTorch/0-indexing
    if emotion == 8: emotion = 0 # surprise is now at 0 index; other emotion indeces unchanged

    # can convert emotion label to emotion string if desired, but
    # training on number is better; better convert to emotion string after predictions are ready
    # emotion = emotions_dict[str(emotion)]

    # get other labels we might want
    intensity = emotion_attributes[file_name.split("-")[3]]

    # get waveform from the sample
    dur.append(librosa.get_duration(filename=file))
    
    waveform, _ = read_audio(file, target_fs = config.sample_rate)

    if actor in train_actor:
        x_train.append(waveform)
        y_train.append(emotion)
        intensity_train.append(intensity)

    elif actor in val_actor:
        x_val.append(waveform)
        y_val.append(emotion)
        intensity_val.append(intensity)

    elif actor in test_actor:
        x_test.append(waveform)
        y_test.append(emotion)
        intensity_test.append(intensity)
    
    file_count += 1
    # keep track of data loader's progress
    if not file_count % 100:
        print('Processed {}/1440 audio samples'.format(file_count))


print('there are {} samples in train, {} samples in val, {} samples in test'.format(len(y_train), len(y_val), len(y_test)))
print('the statis of all audio duration:')
print('median: {}'.format(median(dur)))
print('mean: {}'.format(mean(dur)))
print('stdev: {}'.format(stdev(dur)))
print('percentile: {}'.format(np.percentile(dur, range(0, 105, 5))))

x_train = np.array(x_train, dtype=object)
x_val = np.array(x_val, dtype=object)
x_test = np.array(x_test, dtype=object)

y_train = np.array(y_train, dtype='int32')
y_val = np.array(y_val, dtype='int32')
y_test = np.array(y_test, dtype='int32')

intensity_train = np.array(intensity_train, dtype='S80')
intensity_devel = np.array(intensity_val, dtype='S80')
intensity_test = np.array(intensity_test, dtype='S80')

# save data
dt = h5py.special_dtype(vlen=np.dtype('float32'))
with h5py.File(hdf5_path, 'w') as hf:
	hf.create_dataset("train_audio",  data=x_train, dtype=dt)
	hf.create_dataset("train_y", data=y_train, dtype='int32')
	hf.create_dataset("train_intensity", data=intensity_train, dtype='S80')
	hf.create_dataset("val_audio",  data=x_val, dtype=dt)
	hf.create_dataset("val_y", data=y_val, dtype='int')
	hf.create_dataset("val_intensity", data=intensity_val, dtype='S80')
	hf.create_dataset("test_audio",  data=x_test, dtype=dt)
	hf.create_dataset("test_y", data=y_test, dtype='int')
	hf.create_dataset("test_intensity", data=intensity_test, dtype='S80')
print('Save train, val and test audio arrays to hdf5 located at {}'.format(hdf5_path))

hf.close()
