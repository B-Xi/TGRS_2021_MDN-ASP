import six
import keras as K
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    multiply,
    GlobalMaxPooling3D,
    PReLU
)
from keras.layers.convolutional import Conv3D
from keras.layers.core import Reshape
from keras.regularizers import l2
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def _bn_relu_spc(input):
    norm = BatchNormalization(axis=4)(input)
    return Activation("relu")(norm)

def _conv_bn_relu_spc(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    def f(input):
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer, filters=nb_filter,
                      kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)
        return _bn_relu_spc(conv)
    return f

def _bn_relu_conv_spc(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    def f(input):
        activation = _bn_relu_spc(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(
            activation)
    return f

def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = block_function(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=(is_first_layer and i == 0)
            )(input)
        return input
    return f

def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample,
                           kernel_regularizer=l2(0.0001),
                           filters=nb_filter, kernel_size=(1, 1, 7), padding='same')(input)
        else:
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
                                      subsample=init_subsample)(input)

        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)
        return _shortcut_spc(input, residual)
    return f

def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
     def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample,
                           kernel_regularizer=l2(0.0001),
                           filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1,
                                  subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut(input, residual)
     return f

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def _shortcut_spc(input, residual):
    shortcut = squeeze_excite_block(residual)
    return add([shortcut, input])

def _conv_bn_relu(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    def f(input):
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)
        return _bn_relu(conv)
    return f

def _bn_relu(input):
    norm = BatchNormalization(axis=4)(input)
    return Activation("relu")(norm)

def _bn_relu_conv(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    def f(input):
        activation = _bn_relu(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                      filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(
            activation)
    return f

def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2, 1)
            input = block_function(
                nb_filter=nb_filter,
                init_subsample=init_subsample,
                is_first_block_of_first_layer=(is_first_layer and i == 0)
            )(input)
        return input
    return f

def _shortcut(input, residual):
    shortcut = squeeze_excite_block(residual)
    return add([shortcut, input])

def squeeze_excite_block(input, ratio=8):
    init = input
    channel_axis = 1 if K.backend.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalMaxPooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    if K.backend.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    x = multiply([init, se])
    return x