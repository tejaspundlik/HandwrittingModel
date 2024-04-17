import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model

def activation_layer(
        x: tf.Tensor,
        activation: str)->tf.Tensor:
    if activation == "relu":
        x =layers.ReLU()(x)
    elif activation == "leaky_relu":
        x = layers.LeakyReLU()(x)
    return x
    
def res_block(
        x: tf.Tensor,
        filter_num: int,
        strides: int = 2,
        kernel_size: int = 3,
        skip_conv: bool = True,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        activation: str = "relu",
        dropout: float = 0.2):
    
    x_skip = x
    x = layers.Conv2D(filter_num, kernel_size, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    x = layers.Conv2D(filter_num, kernel_size, padding = padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)

    if skip_conv:
        x_skip = layers.Conv2D(filter_num, 1, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x_skip)

    x = layers.Add()([x, x_skip])     
    x = activation_layer(x, activation=activation)

    if dropout:
        x = layers.Dropout(dropout)(x)

    return x

#above ref: https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = res_block(input, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x2 = res_block(x1, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = res_block(x2, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x4 = res_block(x3, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = res_block(x4, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x6 = res_block(x5, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = res_block(x6, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    squeezed = layers.Reshape((x7.shape[-3] * x7.shape[-2], x7.shape[-1]))(x7)
    blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model
