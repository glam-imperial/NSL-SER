import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], './neural-structured-learning'))
from neural_structured_learning.configs import DistanceType


# dataset parameters
sample_rate = 16000
audio_samples = 95744
num_classes = 7
# emb_shape = (298, 768)
logmel_shape = (373, 64)
logmel_1d = logmel_shape[0] * logmel_shape[1]
audio_1d = 95744

# msp dataset
msp_audio_samples = 8.43 * sample_rate

# ravdess dataset, 4 second is about the 80% of all audio samples
rav_class_num = 8
rav_seq_len = sample_rate * 4


# neural graph learning parameters
# distance_type = nsl.configs.DistanceType.L2 # Could also consider COSINE
# distance_type = nsl.configs.DistanceType.COSINE
distance_type = DistanceType.COSINE
graph_regularization_multiplier = 0.1
num_neighbors = 10

# training parameters
train_epochs = 50
batch_size = 16
