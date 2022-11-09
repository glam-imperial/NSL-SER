import os
import sys
import numpy as np
import argparse
import time
import logging
import config
import pandas as pd
import json
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

sys.path.insert(1, os.path.join(sys.path[0], './neural-structured-learning'))
from neural_structured_learning.configs import make_graph_reg_config
from neural_structured_learning.keras import GraphRegularization
from models_demos import vgg15, MobileNet15, ResNet9, cnn6
import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import recall_score

# wav2vec related
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, TFWav2Vec2Model
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, Wav2Vec2ForPreTraining
from models_msp import Wav2vec2FineTune

NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'

sys.path.insert(1, os.path.join(sys.path[0], './utils'))
from utilities import (read_audio, create_folder, metrics_uar,
                       get_filename, create_logging, calculate_accuracy,
                       print_accuracy, calculate_confusion_matrix,
                       move_data_to_gpu, audio_unify, uar)
'''
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print(
    "GPU is",
    "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
'''
# distribute training
# gpus = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)


default_logmel = np.zeros((config.logmel_1d), dtype=np.float32)
default_weight = np.zeros((1), dtype=np.float32)
default_label = -1 * np.ones((config.num_classes), dtype=np.int64)

def make_dataset(file_path, num_neighbors, training=False):
  """Creates a `tf.data.TFRecordDataset`.

  Args:
    file_path: Name of the file in the `.tfrecord` format containing
      `tf.train.Example` objects.
    training: Boolean indicating if we are in training mode.

  Returns:
    An instance of `tf.data.TFRecordDataset` containing the `tf.train.Example`
    objects.
  """

  def parse_example(example_proto):
    """Extracts relevant fields from the `example_proto`.

    Args:
      example_proto: An instance of `tf.train.Example`.

    Returns:
      A pair whose first value is a dictionary containing relevant features
      and whose second value contains the ground truth labels.
    """
    feature_spec = {
        'logmel': tf.io.FixedLenFeature([config.logmel_1d], np.float32),
        'label': tf.io.FixedLenFeature([config.num_classes], np.int64, default_value=default_label),
    }

    # We also extract corresponding neighbor features in a similar manner to
    # the features above during training.
    if training:
      for i in range(num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'logmel')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i,
                                         NBR_WEIGHT_SUFFIX)
        feature_spec[nbr_feature_key] = tf.io.FixedLenFeature(
                [config.logmel_1d], np.float32,
                default_value=default_logmel)
        # We assign a default value of 0.0 for the neighbor weight so that
        # graph regularization is done on samples based on their exact number
        # of neighbors. In other words, non-existent neighbors are discounted.
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], np.float32, default_value=default_weight)


    features = tf.io.parse_single_example(example_proto, feature_spec)
    '''
    ##### Leave this work in the models part
    # Reshape the 'logmel' feature
    features['logmel'] = tf.reshape(features['logmel'].values, list(config.logmel_shape))
    if training:
      for i in range(config.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'logmel')
        features[nbr_feature_key] = tf.reshape(features[nbr_feature_key].values, list(config.logmel_shape))
    '''
    labels = features.pop('label')
    return features, labels

  dataset = tf.data.TFRecordDataset([file_path])
  if training:
    dataset = dataset.shuffle(1234)
  dataset = dataset.map(parse_example)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  # dataset = dataset.repeat(config.train_epochs)
  return dataset

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr



def train(args):
    multiplier = args.multiplier
    num_neighbors = args.num_neighbors
    target_layer = args.target_layer
    base_model = args.base_model
    validation = args.validation
    workspace = args.workspace

    '''
    # add the weights for the loss
    if validation:
        emo_weights = tf.constant([1000/453, 1000/666, 1000/561, 1000/461, 1000/586, 1000/396, 1000/606])
    else:
        emo_weights = tf.constant([1000/(453+395), 1000/(666+608), 1000/(561+471), 1000/(461+404), 1000/(586+531), 1000/(396+358), 1000/(606+543)])
    '''

    graph_reg_config = make_graph_reg_config(
        max_neighbors=num_neighbors,
        multiplier=multiplier,
        distance_type=config.distance_type,
        sum_over_axis=-1)
    print('target layer is {}, # neighbors is {}, multiplier is {}.'.format(target_layer, num_neighbors, multiplier))

    # Prepare the datasets
    if validation:
        split_keyword = 'val'
    else:
        split_keyword = 'test'
    train_tfr_path = os.path.join(workspace, 'graph_files_rav_made', split_keyword, '{}'.format(num_neighbors), 'nsl_train_data.tfr')
    test_tfr_path = os.path.join(workspace, 'graph_files_rav_made', split_keyword, '{}'.format(num_neighbors), 'test_data.tfr')

    train_dataset = make_dataset(train_tfr_path, num_neighbors, training=True)
    test_dataset = make_dataset(test_tfr_path, num_neighbors, training=False)

    train_num = sum(1 for _ in tf.data.TFRecordDataset(train_tfr_path))
    test_num = sum(1 for _ in tf.data.TFRecordDataset(test_tfr_path))
    print('valiation = {}, # training sampels: {} , # test samples: {}'.format(validation, train_num, test_num))

    if base_model == 'vgg':
        model = vgg15()
    elif base_model == 'resnet':
        model = ResNet9()
    elif base_model == 'mobilenet':
        model = MobileNet15()
    elif base_model == 'cnn':
        model = cnn6()


    print('There are in total of {} parameters in the {}'.format(model.count_params(), base_model))

    # decay every 5/50 epoch
    decay_steps = (train_num // config.batch_size) * 5
    # lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.001, decay_steps=decay_steps)

    # optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=decay_steps, decay_rate=0.9, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lr_metric = get_lr_metric(optimizer)

    # Wrap the base model with graph regularization.
    graph_reg_model = GraphRegularization(model, base_model,  target_layer, config.batch_size, num_neighbors, graph_reg_config)

    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    graph_reg_model.compile(
        optimizer=optimizer,
        loss=loss_func,
        run_eagerly=True,
        metrics=["accuracy", uar, lr_metric])

    '''
    # set the early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_uar', patience=5)

    history = graph_reg_model.fit(
        x=train_dataset,
        validation_data=test_dataset,
        epochs=config.train_epochs,
        # steps_per_epoch=train_num // config.batch_size,
        # validation_steps=3310 // config.batch_size,
        callbacks=[callback],
        verbose=1)
    '''
    if validation:
        # set the early stopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_uar', patience=5, mode='max')
        history = graph_reg_model.fit(
            x=train_dataset,
            validation_data=test_dataset,
            epochs=config.train_epochs,
            # steps_per_epoch=train_num // config.batch_size,
            # validation_steps=3310 // config.batch_size,
            # callbacks=[callback],
            verbose=1)
        filepath = os.path.join(workspace, 'logs_rav_made', 'ravdess_base', 'devel', '{}_{}_{}_{}.csv'.format(base_model, target_layer, multiplier, num_neighbors))
        pd.DataFrame(history.history).to_csv(filepath, sep='\t')

    else:
        history = graph_reg_model.fit(
            x=train_dataset,
            epochs=config.train_epochs,
            verbose=1)

        filepath = os.path.join(workspace, 'logs_rav_made', 'ravdess_base', 'test', '{}_{}_{}_{}.csv'.format(base_model, target_layer, multiplier, num_neighbors))
        pd.DataFrame(history.history).to_csv(filepath, sep='\t')

        test_res = graph_reg_model.evaluate(test_dataset, return_dict=True)
        filepath = os.path.join(workspace, 'logs_rav_made', 'ravdess_base', 'test', '{}_{}_{}_{}.txt'.format(base_model, target_layer, multiplier, num_neighbors))
        with open(filepath, 'w') as file:
            file.write(json.dumps(test_res))

        modelpath = os.path.join(workspace, 'models_rav_made', '{}_{}_{}_{}'.format(base_model, target_layer, multiplier, num_neighbors))
        graph_reg_model.save(modelpath)
        print('Model developed on train+validation is saved to {}'.format(modelpath))

    '''
    # save the model and training history
    filepath = os.path.join(workspace, 'logs_rav_made', 'ravdess_base', '{}_{}_{}_{}_{}.csv'.format(validation, base_model, target_layer, multiplier, num_neighbors))
    pd.DataFrame(history.history).to_csv(filepath, sep='\t')
    modelpath = os.path.join(workspace, 'models_rav_made', '{}_{}_{}_{}_{}'.format(validation, base_model, target_layer, multiplier, num_neighbors))
    graph_reg_model.save(modelpath)
    print('Model saved to {}'.format(modelpath))
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, default='/storage/home/ychang/DEMOS')
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--num_neighbors', type=int, required=True)
    parser_train.add_argument('--target_layer', type=str, required=True)
    parser_train.add_argument('--multiplier', type=float, required=True)
    parser_train.add_argument('--base_model', type=str, default='vgg')

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs_rav_made', args.filename)
    # create_logging(logs_dir, filemode='w')
    print(args)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')
