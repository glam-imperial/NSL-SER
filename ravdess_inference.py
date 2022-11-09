import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import sys
from random import shuffle
from numpy import genfromtxt
import os
pd.set_option('display.max_rows', 500)
import h5py
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
import config
import torch

sys.path.insert(1, os.path.join(sys.path[0], './utils'))
from utilities import (read_audio, create_folder,
                       get_filename, create_logging, calculate_accuracy,
                       print_accuracy, calculate_confusion_matrix,
                       move_data_to_gpu, audio_unify)
import argparse
import time
import logging

# wav2vec related
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, Wav2Vec2ForPreTraining
from models_msp import Wav2vec2FineTune
# For pytorch dataset
from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor
    def __getitem__(self, index):
        return self.data_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

def data_generater(hdf5_path):
    '''Read data into a dict'''
    with h5py.File(hdf5_path, 'r') as hf:
        x_audio_train = hf['train_audio'][:]
        x_audio_dev = hf['dev_audio'][:]
        x_name_train = hf['train_name'][:]
        x_name_dev = hf['dev_name'][:]

    hf.close()

    x_train = np.concatenate((x_audio_train, x_audio_dev), axis=0)
    x_name = np.concatenate((x_name_train, x_name_dev), axis=0)
    d = {'dev':Dataset.from_dict({'audio':x_train})}
    return d, x_name


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
def preprocess_function(examples):
    audio_arrays = [x for x in examples['audio']]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate)
    return inputs



def inference_w2v(args):
    # Some paths
    workspace = args.workspace
    hdf5_path = os.path.join(workspace, 'train_dev_audio.h5')
    emb_dir = os.path.join(workspace, 'embeddings_rav_made')
    model_path = args.model_path

    cuda = args.cuda

    # Load the fine-tuned wav2vec2 model
    # model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
    # model = Wav2vec2FineTune(model, class_num)
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=8)
    model.load_state_dict(torch.load(model_path))
    if cuda:
        model.cuda()
    model.eval()

    # For debug only
    # for name, layer in model.named_modules():
    #    print(name, layer)
    logging.info('To generate embeddings for DEMOS, the loaded fine-tuned wav2vec2 model based on RAVDESS dataset:')
    logging.info(model)

    # Create the Pytorch dataset for inference
    data, x_name = data_generater(hdf5_path)
    demos_dataset = DatasetDict(data)
    demos_dataset = demos_dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=1)
    demos_dataset = TensorDataset(torch.Tensor([audio_unify(x) for x in demos_dataset['dev']['input_values']]))
    demos_dataloader = torch.utils.data.DataLoader(demos_dataset, batch_size=1, shuffle=False)


    # Inference time
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.projector.register_forward_hook(get_activation('projector'))
    # demos_dataset:'input_values' <class 'list'> 44549
    # for key, value in demos_dataset['train'][0].items():
    for (i, batch_x) in enumerate(demos_dataloader):
        activation = {}
        batch_x = move_data_to_gpu(batch_x, cuda=cuda)
        output = model(batch_x)
        emb = activation['projector'].detach().cpu().numpy().squeeze()
        df = pd.DataFrame(emb)
        df.to_csv(os.path.join(emb_dir, x_name[i].decode()+'.csv'), sep='\t', index=False)
        '''
        test_input = demos_dataset['train'][i]
        with torch.no_grad():
            input_values = torch.tensor(test_input["input_values"].cuda()).unsqueeze(0)
            output = model(input_values)
            emb = activation['linear'].cpu().numpy().squeeze()
            # assert emb.shape == (138, 256), "the embedding shape is not as expected"
            df = pd.DataFrame(emb)
            df.to_csv(os.path.join(emb_dir, x_name_train[i].decode()+'.csv'), sep='\t', index=False)
        '''
    logging.info('In total of {} csv files, and the generated embedding per each smaple with dimension: {}'.format(i+1, emb.shape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('inference_w2v')
    parser_train.add_argument('--workspace', type=str, default='/storage/home/ychang/DEMOS')
    parser_train.add_argument('--model_path', type=str, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs_rav_made', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'inference_w2v':
        inference_w2v(args)
    else:
        raise Exception('Error argument!')
