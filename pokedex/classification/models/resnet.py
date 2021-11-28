"""
Python implementation of DNN ResNet, using Keras 2.4.0 and Tensorflow 2.3.0.
Adapted from https://github.com/raghakot/keras-resnet.
"""
import typing as t

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Activation,
    Flatten,
    MaxPooling2D,
    Conv2D,
    Layer,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Add,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
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


def _bn_relu(input: Layer) -> Layer:
    """
    Builds BN -> relu block.
    :param input: input layer
    :return:
    """
    norm = BatchNormalization()(input)
    return Activation('relu')(norm)


def _conv_bn_relu(**conv_param) -> t.Callable:
    """
    Builds conv -> BN -> relu block.
    :return:
    """
    filters = conv_param['filters']
    kernel_size = conv_param['kernel_size']
    strides = conv_param.setdefault('strides', (1, 1))
    kernel_initializer = conv_param.setdefault('kernel_initializer', 'he_normal')
    padding = conv_param.setdefault('padding', 'same')
    kernel_regularizer = conv_param.setdefault('kernel_regularizer', l2(0.0001))

    def f(input: Layer) -> Layer:
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_param) -> t.Callable:
    """
    Builds BN -> relu -> conv block.
    It follows the improvement found in this paper: https://arxiv.org/pdf/1603.05027v2.pdf.
    :param conv_params:
    :return:
    """
    filters = conv_param['filters']
    kernel_size = conv_param['kernel_size']
    strides = conv_param.setdefault('strides', (1, 1))
    kernel_initializer = conv_param.setdefault('kernel_initializer', 'he_normal')
    padding = conv_param.setdefault('padding', 'same')
    kernel_regularizer = conv_param.setdefault('kernel_regularizer', l2(0.0001))

    def f(input: Layer) -> Layer:
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _relu_conv_bn(**conv_param) -> t.Callable:
    """
    Builds relu -> conv -> BN block.
    :param conv_param:
    :return:
    """
    filters = conv_param['filters']
    kernel_size = conv_param['kernel_size']
    strides = conv_param.setdefault('strides', (1, 1))
    kernel_initializer = conv_param.setdefault('kernel_initializer', 'he_normal')
    padding = conv_param.setdefault('padding', 'same')
    kernel_regularizer = conv_param.setdefault('kernel_regularizer', l2(0.0001))

    def f(input: Layer) -> Layer:
        activation = Activation('relu')(input)
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)
        return BatchNormalization()(conv)

    return f


def _shortcut(input: Layer, residual: Layer) -> Layer:
    """
    Makes the link between the input and the residual block and merges them. If both shapes are equals, we use the
    identity ; else, we use a 1x1 convolution before.
    :param input:
    :param residual:
    :return: sum of the residual block and the input
    """
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(0.0001))(input)
    return Add()([shortcut, residual])


def _residual_block(block_function, filters, repetitions: int, is_first_layer: bool = False) -> t.Callable:
    """

    :param block_function:
    :param filters:
    :param repetitions:
    :param is_first_layer:
    :return:
    """

    def f(input: Layer) -> Layer:

        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters,
                                   init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)

        return input

    return f


def full_preactivation_basic_block(filters: int, init_strides: tuple = (1, 1),
                                   is_first_block_of_first_layer: bool = False, **kwargs) -> t.Callable:
    """
    Builds a basic 3x3 convolution block for models with layers <= 34.
    It follows the improvement found in this paper: https://arxiv.org/pdf/1603.05027v2.pdf.
    :param filters:
    :param init_strides:
    :param is_first_block_of_first_layer:
    :return:
    """

    def f(input: Layer) -> Layer:

        if is_first_block_of_first_layer:
            # since we just did conv -> bn -> relu -> maxpool, there is no need to do bn -> relu again
            conv = Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          strides=init_strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(0.0001))(input)
        else:
            conv = _bn_relu_conv(filters=filters,
                                 kernel_size=(3, 3),
                                 strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv)
        return _shortcut(input, residual)

    return f


def full_preactivation_bottleneck(filters: int, init_strides: tuple = (1, 1),
                                  is_first_block_of_first_layer: bool = False, **kwargs) -> t.Callable:
    """
    Bottleneck for models with more than 34 layers.
    It follows the improvement found in this paper: https://arxiv.org/pdf/1603.05027v2.pdf.
    :return:
    """

    def f(input: Layer) -> Layer:

        if is_first_block_of_first_layer:
            # since we just did conv -> bn -> relu -> maxpool, there is no need to do bn -> relu again
            conv1_1 = Conv2D(filters=filters,
                             kernel_size=(1, 1),
                             strides=init_strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(0.0001))(input)
        else:
            conv1_1 = _bn_relu_conv(filters=filters,
                                    kernel_size=(1, 1),
                                    strides=init_strides)(input)

        conv3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv3_3)
        return _shortcut(input, residual)

    return f


def _get_block(identifier):
    if isinstance(identifier, str):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def build_resnet() -> Model:
    raw_model: t.Dict = config.get("models")
    layers = raw_model["layers"]
    # Model parameters
    img_channels = raw_model["img_channels"]
    img_rows = raw_model["img_rows"]
    img_cols = raw_model["img_cols"]
    nb_classes = raw_model["nb_classes"]

    if layers == 18:
        return ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
    if layers == 34:
        return ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
    if layers == 50:
        return ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
    if layers == 101:
        return ResnetBuilder.build_resnet_101((img_channels, img_rows, img_cols), nb_classes)
    if layers == 152:
        return ResnetBuilder.build_resnet_152((img_channels, img_rows, img_cols), nb_classes)

    raise RuntimeError(f"Unknown number of layers: {layers}.")


class ResnetBuilder:
    @staticmethod
    def build(input_shape: tuple, num_outputs: int, block_fn, repetitions: t.List) -> Model:
        """

        :param input_shape:
        :param num_outputs:
        :param block_fn:
        :param repetitions:
        :return:
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)

        # first layer is a 7x7 convolution followed by a bn -> relu -> maxpool
        conv7_7 = _conv_bn_relu(filters=64,
                                kernel_size=(7, 7),
                                strides=(2, 2))(input)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv7_7)

        filters = 64
        block = maxpool
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        block = _bn_relu(block)

        # classifier
        block_shape = K.int_shape(block)

        final_pool = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                      strides=(1, 1))(block)
        flatten = Flatten()(final_pool)
        dense = Dense(units=num_outputs,
                      kernel_initializer='he_normal',
                      activation='softmax')(flatten)
        model = Model(inputs=input, outputs=dense)
        return model

    # 18-layer classification : no bottleneck
    @staticmethod
    def build_resnet_18(input_shape: tuple, num_outputs: int) -> Model:
        basic_block = config.get("models", "basic_block")
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    # 34-layer classification : no bottleneck
    @staticmethod
    def build_resnet_34(input_shape: tuple, num_outputs: int) -> Model:
        basic_block = config.get("models", "basic_block")
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    # 50-layer classification : bottleneck
    @staticmethod
    def build_resnet_50(input_shape: tuple, num_outputs: int) -> Model:
        basic_block = config.get("models", "basic_block")
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    # 101-layer classification : bottleneck
    @staticmethod
    def build_resnet_101(input_shape: tuple, num_outputs: int) -> Model:
        bottleneck = config.get("models", "bottleneck")
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    # 152-layer classification : bottleneck
    @staticmethod
    def build_resnet_152(input_shape: tuple, num_outputs: int) -> Model:
        bottleneck = config.get("models", "bottleneck")
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
