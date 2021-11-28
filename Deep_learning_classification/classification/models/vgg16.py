import typing as t

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tensorflow.keras import backend as K

from .. import config


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        """
        Image data is represented in a three-dimensional array where the last channel represents the color channels
        """
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        """
        Image data is represented in a three-dimensional array where the first channel represents the color channels
        """
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


class VGG16:
    @staticmethod
    def build_vgg16() -> Sequential:
        """
        Function that generates our DL model for image classification.

        :return: VGG16 model
        """
        # Model parameters
        raw_model: t.Dict = config.get("models")
        dropout = raw_model["dropout"]
        activation = raw_model["activation"]
        final_activation = raw_model["final_activation"]
        padding = raw_model["padding"]

        img_channels = raw_model["img_channels"]
        img_rows = raw_model["img_rows"]
        img_cols = raw_model["img_cols"]
        input_shape = (img_channels, img_rows, img_cols)

        nb_classes = raw_model["nb_classes"]

        _handle_dim_ordering()

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
            print("\nCHANNELS\n")
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
            print(f'INPUT SHAPE : {input_shape}')

        model = tf.keras.models.Sequential([
            # First block
            Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   input_shape=input_shape,
                   name='block1_conv1'),
            Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block1_conv2'),
            MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2), name='block1_pool'),

            # Second block
            Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block2_conv1'),
            Conv2D(filters=128,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block2_conv2'),
            MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2), name='block2_pool'),

            # Third block
            Conv2D(filters=256,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block3_conv1'),
            Conv2D(filters=256,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block3_conv2'),
            Conv2D(filters=256,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block3_conv3'),
            MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2), name='block3_pool'),

            # Fourth block
            Conv2D(filters=512,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block4_conv1'),
            Conv2D(filters=512,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block4_conv2'),
            Conv2D(filters=512,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block4_conv3'),
            MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2), name='block4_pool'),

            # Fifth block
            Conv2D(filters=512,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block5_conv1'),
            Conv2D(filters=512,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block5_conv2'),
            Conv2D(filters=512,
                   kernel_size=(3, 3),
                   padding=padding,
                   activation=activation,
                   name='block5_conv3'),
            MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2), name='block5_pool'),

            # Classification block
            Flatten(name='flatten'),

            Dense(units=4096,
                  activation=activation, name='fc1'),
            Dropout(dropout),

            Dense(units=4096,
                  activation=activation, name='fc2'),
            Dropout(dropout),

            Dense(units=nb_classes,
                  activation=final_activation, name='pred')
        ])
        return model
