#!/bin/bash

# Split the ravdess dataset 
python utils/ravdess_data_split.py

# Fine-tune the pre-trained wav2vec 2.0 model based on the ravdess dataset
python ravdess_finetune.py train --validation --epoch 20 --cuda --freeze 
python ravdess_finetune.py train --epoch 20 --cuda --freeze 

# Generate the embeddings for ravdess dataset
python ravdess_inference.py inference_w2v --cuda

# Generate the base log Mel spectrograms and graph for the demos dataset
python ravdess_features.py generate_features --num_neighbors 3 --validation 
python ravdess_features.py generate_features --num_neighbors 6 --validation 
python ravdess_features.py generate_features --num_neighbors 9 --validation 

python ravdess_features.py generate_features --num_neighbors 3 
python ravdess_features.py generate_features --num_neighbors 6 
python ravdess_features.py generate_features --num_neighbors 9 



# VGG model development

python main.py train --target_layer 'fc1' --num_neighbors 3 --multiplier 0.1 --base_model 'vgg15' 

python main.py train --target_layer 'fc1' --num_neighbors 6 --multiplier 0.1 --base_model 'vgg15' 

python main.py train --target_layer 'fc1' --num_neighbors 9 --multiplier 0.1 --base_model 'vgg15'
