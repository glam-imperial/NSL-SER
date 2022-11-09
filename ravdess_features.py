import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# import neural_structured_learning as nsl
sys.path.insert(1, os.path.join(sys.path[0], './neural-structured-learning'))
from neural_structured_learning.tools import pack_nbrs, build_graph_from_config
from neural_structured_learning.configs import GraphBuilderConfig
import tensorflow as tf
import pandas as pd
import math
from random import shuffle
from numpy import genfromtxt
pd.set_option('display.max_rows', 500)
import h5py
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize

sys.path.insert(1, os.path.join(sys.path[0], './utils'))
from utilities import (read_audio, create_folder,
                       get_filename, create_logging, calculate_accuracy,
                       print_accuracy, calculate_confusion_matrix,
                       move_data_to_gpu, audio_unify)

import argparse
import time
import logging
import config
emos = ['col', 'dis', 'gio', 'pau', 'rab', 'sor', 'tri']
PATH_TO_HDF5_FILE = '/storage/home/ychang/DEMOS/logmel_demos.hdf5'


def _int64_feature(value):
  """Returns int64 tf.train.Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def _bytes_feature(value):
  """Returns bytes tf.train.Feature."""
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def _float_feature(value):
  """Returns float tf.train.Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))


# Prepare the log Mel spectrograms as features
def calculate_scalar(x):
    if x.ndim == 2:
        asix = 0
    elif x.ndim ==3:
        axis = (0, 1)
    mean_val = np.mean(x, axis=axis)
    std_val = np.std(x, axis=axis)
    return mean_val, std_val

def scale(x, mean_val, std_val):
    return (x - mean_val) / std_val

with h5py.File(PATH_TO_HDF5_FILE, "r") as f:
    a_group_key = list(f.keys())
    data = pd.DataFrame()
    for key in a_group_key:
        data[key] = np.array(f[key]).tolist()

# Decoding
for column in data.columns:
    if column != 'logmel':
       data[column] = data[column].apply(lambda x: x.decode('UTF-8'))

def generate_features(args):
    workspace = args.workspace
    num_neighbors = args.num_neighbors
    validation = args.validation

    # Some paths
    emb_dir = os.path.join(workspace, 'embeddings_rav_made')
    graph_dir = os.path.join(workspace, 'graph_files_rav_made')
    div_dir = '/home/ychang/nsl23/workspace/demos_split'

    train_list = np.squeeze(pd.read_csv(os.path.join(div_dir, 'train.csv'), header=None).to_numpy())
    dev_list = np.squeeze(pd.read_csv(os.path.join(div_dir, 'dev.csv'), header=None).to_numpy())
    test_list = np.squeeze(pd.read_csv(os.path.join(div_dir, 'test.csv'), header=None).to_numpy())
    # logging.info("Version: ", tf.__version__)
    # logging.info("Eager mode: ", tf.executing_eagerly())
    # print("Hub version: ", hub.__version__)
    # logging.info(
    #             "GPU is",
    #             "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
    logging.info('original # samples in DEMOS train/dev/test is {}/{}/{}'.format(train_list.shape[0], dev_list.shape[0], test_list.shape[0]))

    train_emb_label = data[data['audio_name'].isin(train_list)]
    dev_emb_label = data[data['audio_name'].isin(dev_list)]
    test_emb_label = data[data['audio_name'].isin(test_list)]

    ##### Prepare the train and test data sets, return x_train, y_train, x_test, y_test #######
    if validation:
        # Scale the logmel based on the train set only
        train_logmel = np.array([x for x in train_emb_label.logmel])
        mean_val, std_val = calculate_scalar(train_logmel)

        # Train data is the train set
        x_train = np.array([x for x in train_emb_label.logmel])
        x_train = scale(x_train, mean_val, std_val)
        x_train = x_train[:,:,:,np.newaxis]
        # {ndarray, sparse matrix} of shape (n_samples, n_classes)
        y_train = label_binarize([x for x in train_emb_label.emotion], classes=emos)

        # Evaluation data is the dev set
        x_test = np.array([x for x in dev_emb_label.logmel])
        x_test = scale(x_test, mean_val, std_val)
        x_test = x_test[:,:,:,np.newaxis]
        y_test = label_binarize([x for x in dev_emb_label.emotion], classes=emos)
    else:
        # Scale the logmel based on the train+dev set
        train_logmel = [x for x in train_emb_label.logmel]
        dev_logmel = [x for x in dev_emb_label.logmel]
        train_logmel.extend(dev_logmel)
        train_logmel = np.array(train_logmel)
        mean_val, std_val = calculate_scalar(train_logmel)

        y_train = [x for x in train_emb_label.emotion]
        y_dev = [x for x in dev_emb_label.emotion]
        y_train.extend(y_dev)

        # Train data is the train + dev sets
        x_train = scale(train_logmel, mean_val, std_val)
        x_train = x_train[:,:,:,np.newaxis]
        # {ndarray, sparse matrix} of shape (n_samples, n_classes)
        y_train = label_binarize(y_train, classes=emos)

        # Evaluation data is the test set
        x_test = np.array([x for x in test_emb_label.logmel])
        x_test = scale(x_test, mean_val, std_val)
        x_test = x_test[:,:,:,np.newaxis]
        y_test = label_binarize([x for x in test_emb_label.emotion], classes=emos)


    def create_embedding_example(audio_id, record_id):
      """Create tf.Example containing the sample's embedding and its ID."""

      emb_path = os.path.join(emb_dir, audio_id + '.csv')
      # emb_path = os.path.join(emb_dir, str(record_id) + '.csv')
      emb = pd.read_csv(emb_path, sep='\t')

      # Flatten the sentence embedding back to 1-D.
      # sentence_embedding = tf.reshape(sentence_embedding, shape=[-1])

      # Flatten the embedding (138, 768) into 1-D
      # emb = emb.to_numpy().flatten()
      emb = emb.to_numpy()
      emb = emb.mean(axis=0)
      if record_id == 1:
          logging.info('emb shape is: {}'.format(emb.shape))
      # emb = emb.to_numpy()
      # emb = emb.mean(axis=0)
      features = {
          # 'id': _bytes_feature(str(record_id)),
          'id': _bytes_feature(str(audio_id)),
          'embedding': _float_feature(emb)
      }
      return tf.train.Example(features=tf.train.Features(feature=features))

    def create_embeddings(ids, output_path, starting_record_id):
      start_id = int(starting_record_id)
      with tf.io.TFRecordWriter(output_path) as writer:
        for i, audio_id in enumerate(ids):
            record_id = start_id + i
            example = create_embedding_example(audio_id, record_id)
            writer.write(example.SerializeToString())
      return record_id

    # Persist TF.Example features containing embeddings for training data in TFRecord format.
    if validation:
        output_path_tfr = os.path.join(graph_dir, 'val', '{}'.format(num_neighbors), 'train_embeddings.tfr')
        output_path_tsv = os.path.join(graph_dir, 'val', '{}'.format(num_neighbors), 'graph.tsv')
        test_list = dev_list
    else:
        output_path_tfr = os.path.join(graph_dir, 'test', '{}'.format(num_neighbors), 'train_embeddings.tfr')
        output_path_tsv = os.path.join(graph_dir, 'test', '{}'.format(num_neighbors), 'graph.tsv')
        train_list = np.concatenate((train_list, dev_list), axis=None)
        test_list = dev_list
    logging.info('Since validation = {}'.format(validation))
    logging.info('current train size: {} and test size: {}'.format(train_list.shape[0], test_list.shape[0]))

    create_embeddings(train_list, output_path_tfr, 0)
    # Build the Graph
    graph_builder_config = GraphBuilderConfig(similarity_threshold=0.99, lsh_splits=2, lsh_rounds=4, random_seed=12345)
    build_graph_from_config([output_path_tfr],
                            output_path_tsv,
                            graph_builder_config)
    logging.info('The graph has been built.')


    # Construct logmel features
    def create_example(logmel, label, record_id, purpose):
      """Create tf.Example containing the sample's word vector, label, and ID."""
      # Reshape the logmel (373, 64, 1) to 1-d array, the shape of label (7,)
      logmel = np.reshape(logmel, -1)
      if purpose == 'train':
          idlist = train_list
      elif purpose == 'test':
          idlist = test_list
      features = {
          # 'id': _bytes_feature(str(record_id)),
          'id': _bytes_feature(idlist[record_id]),
          'logmel': _float_feature(logmel),
          'label': _int64_feature(np.asarray(label)),
      }
      return tf.train.Example(features=tf.train.Features(feature=features))

    def create_records(div, labels, record_path, purpose, starting_record_id):
      start_id = int(starting_record_id)
      with tf.io.TFRecordWriter(record_path) as writer:
        for i, (logmel, label) in enumerate(zip(div, labels)):
            record_id = start_id + i
            example = create_example(logmel, label, record_id, purpose)
            writer.write(example.SerializeToString())
      return record_id


    # Persist TF.Example features (word vectors and labels) for training and dev data in TFRecord format.
    output_train_tfr = os.path.join(os.path.dirname(output_path_tfr), 'train_data.tfr')
    next_record_id = create_records(x_train, y_train, output_train_tfr, 'train', 0)
    output_test_tfr = os.path.join(os.path.dirname(output_path_tfr), 'test_data.tfr')
    if not os.path.isfile(output_test_tfr):
        test_record_id = create_records(x_test, y_test, output_test_tfr, 'test', 0)

    output_nsl_train_tfr = os.path.join(os.path.dirname(output_path_tfr), 'nsl_train_data.tfr')
    pack_nbrs(
        output_train_tfr,
        '',
        output_path_tsv,
        output_nsl_train_tfr,
        add_undirected_edges=True,
        max_nbrs=num_neighbors+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('generate_features')
    parser_train.add_argument('--workspace', type=str, default='/storage/home/ychang/DEMOS')
    parser_train.add_argument('--num_neighbors', type=int, required=True)
    parser_train.add_argument('--validation', action='store_true', default=False)
    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs_rav_made', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'generate_features':
        generate_features(args)
    else:
        raise Exception('Error argument!')
