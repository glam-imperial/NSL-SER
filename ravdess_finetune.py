import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
import librosa
from random import shuffle
import math
from numpy import genfromtxt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os, glob
pd.set_option('display.max_rows', 500)
import h5py
import pickle
from sklearn import preprocessing
import argparse
import logging
from sklearn.preprocessing import label_binarize
from statistics import mean, variance, median
from collections import Counter
import config

sys.path.insert(1, os.path.join(sys.path[0], './utils'))
from utilities import (read_audio, create_folder,
                       get_filename, create_logging, calculate_accuracy,
                       print_accuracy, calculate_confusion_matrix,
                       move_data_to_gpu, audio_unify)
# wav2vec related
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, Wav2Vec2ForPreTraining
from transformers import get_scheduler

# For pytorch dataset
from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric

metric = load_metric("recall")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
batch_size = config.batch_size
class_num = config.rav_class_num

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

def data_generater(hdf5_path, validation):
    '''Read data into a dict'''
    with h5py.File(hdf5_path, 'r') as hf:
        x_train = hf['train_audio'][:]
        y_train = hf['train_y'][:]
        x_val = hf['val_audio'][:]
        y_val = hf['val_y'][:]
        x_test = hf['test_audio'][:]
        y_test = hf['test_y'][:]

    hf.close()

    if validation:
        d = {'train':Dataset.from_dict({'label':y_train,'audio':x_train}), 'test':Dataset.from_dict({'label':y_val,'audio':x_val})}
    else:
        x_train = np.concatenate((x_train, x_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        d = {'train':Dataset.from_dict({'label':y_train,'audio':x_train}), 'test':Dataset.from_dict({'label':y_test,'audio':x_test})}
    return d


# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
def preprocess_function(examples):
    audio_arrays = [x for x in examples['audio']]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=config.sample_rate)
    return inputs

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # labels = np.argmax(labels, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')



def evaluate_finetune(model, data_loader, cuda):
    """Evaluate

    Returns:
      accuracy: float
    """

    outputs, targets= forward_finetune(model, data_loader, cuda)

    # loss
    loss_fct = nn.CrossEntropyLoss()
    loss = float(loss_fct(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy())

    # UAR
    classes_num = outputs.shape[-1]
    predictions = np.argmax(outputs, axis=-1)
    acc, uar = calculate_accuracy(targets, predictions, classes_num)

    return loss, acc, uar


def forward_finetune(model, data_loader, cuda):

    outputs = []
    targets = []

    for (idx, (batch_x, batch_y)) in enumerate(data_loader, 0):

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model.eval()
        # [0] to get the logits from SequenceClassifierOutput class
        batch_output = model(batch_x)[0]

        outputs.append(batch_output.data.cpu().numpy())
        targets.append(batch_y.data.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)

    return outputs, targets

def train(args):

    # Arugments & parameters
    workspace = args.workspace
    validation = args.validation
    epoch = args.epoch
    cuda = args.cuda
    freeze = args.freeze

    hdf5_path = os.path.join(workspace, "ravdess.h5")

    if validation and freeze:
        models_dir = os.path.join(workspace, 'models', 'freeze', 'train_devel')

    elif not validation and freeze:
        models_dir = os.path.join(workspace, 'models', 'freeze', 'traindevel_test')

    elif validation and not freeze:
        models_dir = os.path.join(workspace, 'models', 'no_freeze', 'train_devel')

    elif not validation and not freeze:
        models_dir = os.path.join(workspace, 'models', 'no_freeze', 'traindevel_test')

    create_folder(models_dir)

    # data
    data = data_generater(hdf5_path, validation)
    dataset = DatasetDict(data)
    dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)

    # model loading
    # model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", return_dict=False)
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=class_num)
    # Freeze the CNN layers
    if freeze:
        model.freeze_feature_extractor()

    # calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Params: {}".format(total_params))


    # model = Model(model_deep, classes_num)
    # model_summary(model, logging)
    # model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=classes_num)

    if cuda:
        model.cuda()

    # unify data
    logging.info('Data unifying')
    dataset_train = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(config.rav_seq_len)) for x in dataset['train']['input_values']]), torch.LongTensor(dataset['train']['label']))
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_test = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(config.rav_seq_len)) for x in dataset['test']['input_values']]), torch.LongTensor(dataset['test']['label']))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    del data
    del dataset

    # training
    print('Start training ...')
    lr = 3e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    # scheduler = get_scheduler('linear', optimizer, num_warmup_steps=int(0.1 * len(trainloader) * epoch), num_training_steps=int(len(trainloader)*epoch))
    loss_fct = nn.CrossEntropyLoss()

    # Only save the best model at the end of training
    best_uar = 0
    best_epoch = 0
    previous_out_path = os.path.join(models_dir, '1111.pt')

    for epoch_idx in range(0, epoch):
        logging.info('epoch: {}'.format(epoch_idx))
        for (idx, (batch_x, batch_y)) in enumerate(trainloader, 0):
            batch_x = move_data_to_gpu(batch_x, cuda)
            batch_y = move_data_to_gpu(batch_y, cuda)

            model.train()
            batch_output = model(batch_x)[0]

            loss = loss_fct(batch_output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        tr_loss, tr_acc, tr_uar = evaluate_finetune(model, trainloader, cuda)
        te_loss, te_acc, te_uar = evaluate_finetune(model, testloader, cuda)
        logging.info('In Epoch: {}, train_acc: {:.3f}, train_uar: {:.3f}, train_loss: {:.3f}'.format(epoch_idx, tr_acc, tr_uar, tr_loss))
        logging.info('In Epoch: {}, test_acc:{:.3f}, test_uar: {:.3f}, test_loss: {:.3f}'.format(epoch_idx, te_acc, te_uar, te_loss))

        # save model
        '''
        if te_uar > best_uar:
            if os.path.exists(previous_out_path):
                os.remove(previous_out_path)
            best_uar = te_uar
            best_epoch = epoch_idx
            logging.info('Best model found at epoch {} with test_uar {}'.format(best_epoch, best_uar))

            save_out_path = os.path.join(models_dir, "{}_epoch_{}_testuar.pt".format(best_epoch, best_uar))
            torch.save(model.state_dict(), save_out_path)
            previous_out_path = save_out_path
            logging.info('Model saved to {}'.format(save_out_path))
        '''
        if epoch_idx == epoch -1:
            save_out_path = os.path.join(models_dir, "{}_epoch_{:.4f}_{:.4f}.pt".format(epoch_idx, te_acc, te_uar))
            torch.save(model.state_dict(), save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))
    logging.info('finished training')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, default='/storage/home/ychang/RAVDESS')
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--freeze', action='store_true', default=False)


    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')
