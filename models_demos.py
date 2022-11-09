from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, ReLU, Dropout, DepthwiseConv2D
from keras.utils.layer_utils import get_source_inputs
from keras import backend as K
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
import random
import config
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from math import cos,pi
# disable_eager_execution()


def vgg15():
    # Reshape the input
    img_input = Input(shape=(config.logmel_shape[0] * config.logmel_shape[1], ), dtype='float32', name='logmel')
    x = Reshape((config.logmel_shape[0], config.logmel_shape[1], 1))(img_input)
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv1_1')(x)
    x = BatchNormalization(name='conv1_1_bn')(x)
    x = Activation('relu', name='conv1_1_relu')(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='conv1_2')(x)
    x = BatchNormalization(name='conv1_2_bn')(x)
    x = Activation('relu', name='conv1_2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    # Block 2
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='conv2_1')(x)
    x = BatchNormalization(name='conv2_1_bn')(x)
    x = Activation('relu', name='conv2_1_relu')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='conv2_2')(x)
    x = BatchNormalization(name='conv2_2_bn')(x)
    x = Activation('relu', name='conv2_2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    # Block 3
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='conv3_1')(x)
    x = BatchNormalization(name='conv3_1_bn')(x)
    x = Activation('relu', name='conv3_1_relu')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='conv3_2')(x)
    x = BatchNormalization(name='conv3_2_bn')(x)
    x = Activation('relu', name='conv3_2_relu')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='conv3_3')(x)
    x = BatchNormalization(name='conv3_3_bn')(x)
    x = Activation('relu', name='conv3_3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    # Block 4
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv4_1')(x)
    x = BatchNormalization(name='conv4_1_bn')(x)
    x = Activation('relu', name='conv4_1_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv4_2')(x)
    x = BatchNormalization(name='conv4_2_bn')(x)
    x = Activation('relu', name='conv4_2_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv4_3')(x)
    x = BatchNormalization(name='conv4_3_bn')(x)
    x = Activation('relu', name='conv4_3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    # Block 5
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv5_1')(x)
    x = BatchNormalization(name='conv5_1_bn')(x)
    x = Activation('relu', name='conv5_1_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv5_2')(x)
    x = BatchNormalization(name='conv5_2_bn')(x)
    x = Activation('relu', name='conv5_2_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', use_bias=False, name='conv5_3')(x)
    x = BatchNormalization(name='conv5_3_bn')(x)
    x = Activation('relu', name='conv5_3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    # Classification block
    x = GlobalAveragePooling2D(name='pool6')(x)
    x = Dense(256, name='fc1')(x)
    x = Activation('relu', name='fc1/relu')(x)
    x = Dense(config.num_classes, name='fc2')(x)
    # x = Activation('softmax', name='fc2/softmax')(x)
    model = Model(img_input, x, name='vgg15')
    return model


def convolutional_block(x, filter_size, stride):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter_size, (3,3), padding = 'same', strides = stride, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter_size, (3,3), padding = 'same', strides=(1,1), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter_size, (1,1), strides = stride, use_bias=False)(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x


def ResNet9():
    # Step 1 (Setup Input Layer)
    img_input = Input(shape=(config.logmel_shape[0] * config.logmel_shape[1], ), dtype='float32', name='logmel')
    x = Reshape((config.logmel_shape[0], config.logmel_shape[1], 1))(img_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Step 3 Add the Resnet Blocks
    x = convolutional_block(x, 128, 2)
    x = convolutional_block(x, 256, 2)
    x = convolutional_block(x, 512, 2)
    # Step 4 End Dense Network
    # x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    # x = tf.keras.layers.Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation = 'relu', name='fc1')(x)
    x = tf.keras.layers.Dense(config.num_classes, name='fc2')(x)
    model = tf.keras.models.Model(inputs = img_input, outputs = x, name = "ResNet9")
    return model



def mobilnet_block(x, filters, strides):
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = filters, kernel_size = 1, strides = 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def MobileNet15():
    # Step 1 (Setup Input Layer)
    img_input = Input(shape=(config.logmel_shape[0] * config.logmel_shape[1], ), dtype='float32', name='logmel')
    x = Reshape((config.logmel_shape[0], config.logmel_shape[1], 1))(img_input)
    x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Add mobilenet blocks
    x = mobilnet_block(x, filters = 128, strides = 1)
    x = mobilnet_block(x, filters = 128, strides = 2)
    x = mobilnet_block(x, filters = 256, strides = 1)
    x = mobilnet_block(x, filters = 256, strides = 2)
    x = mobilnet_block(x, filters = 512, strides = 1)
    x = mobilnet_block(x, filters = 512, strides = 2)

    # Step 4 End Dense Network
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation = 'relu', name='fc1')(x)
    x = tf.keras.layers.Dense(config.num_classes, name='fc2')(x)
    model = tf.keras.models.Model(inputs = img_input, outputs = x, name = "MobileNet15")
    return model



def cnn6():
    # Reshape the input
    img_input = Input(shape=(config.logmel_shape[0] * config.logmel_shape[1], ), dtype='float32', name='logmel')
    x = Reshape((config.logmel_shape[0], config.logmel_shape[1], 1))(img_input)
    x = Conv2D(64, (5, 5), padding='same', use_bias=False, name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(128, (5, 5), padding='same', use_bias=False, name='conv2')(x)
    x = BatchNormalization(name='conv2_bn')(x)
    x = Activation('relu', name='conv2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    x = Conv2D(256, (5, 5), padding='same', use_bias=False, name='conv3')(x)
    x = BatchNormalization(name='conv3_bn')(x)
    x = Activation('relu', name='conv3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    x = Conv2D(512, (5, 5), padding='same', use_bias=False, name='conv4')(x)
    x = BatchNormalization(name='conv4_bn')(x)
    x = Activation('relu', name='conv4_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Classification block
    x = GlobalAveragePooling2D(name='pool5')(x)
    # x = Dropout(0.2, name='dropout1')(x)
    x = Dense(256, name='fc1')(x)
    x = Activation('relu', name='fc1/relu')(x)
    x = Dense(config.num_classes, name='fc2')(x)
    # x = Activtion('softmax', name='fc2/softmax')(x)
    model = Model(img_input, x, name='cnn6')
    return model
