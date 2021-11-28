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


class AlexNet:
    @staticmethod
    def build_alexnet() -> Sequential:
        """
        Function that generates our DL model for image classification.

        :return: AlexNet model
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
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        model = tf.keras.models.Sequential([
            Conv2D(filters=96,
                   kernel_size=(11, 11),
                   strides=(4, 4),
                   activation=activation,
                   input_shape=input_shape,
                   padding=padding),
            BatchNormalization(),
            MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2)),

            Conv2D(filters=256,
                   kernel_size=(5, 5),
                   strides=(1, 1),
                   activation=activation,
                   padding=padding),
            BatchNormalization(),
            MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2)),

            Conv2D(filters=384,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=activation,
                   padding=padding),

            Conv2D(filters=384,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=activation,
                   padding=padding),

            Conv2D(filters=256,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=activation,
                   padding=padding),
            BatchNormalization(),
            MaxPooling2D(pool_size=(3, 3),
                         strides=(2, 2)),

            Flatten(),

            Dense(units=4096,
                  activation=activation),
            Dropout(dropout),

            Dense(units=4096,
                  activation=activation),
            Dropout(dropout),

            Dense(units=nb_classes,
                  activation=final_activation)
        ])
        return model
