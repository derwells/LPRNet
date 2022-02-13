import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def conv2D_batchnorm(*args, **kwargs):
    "2d "
    return keras.Sequential(
        [layers.Conv2D(*args, **kwargs), layers.BatchNormalization(), layers.ReLU()]
    )


def basic_block(channel_out):
    return keras.Sequential(
        [
            conv2D_batchnorm(channel_out // 4, (1, 1), padding="same"),
            conv2D_batchnorm(channel_out // 4, (3, 1), padding="same"),
            conv2D_batchnorm(channel_out // 4, (1, 3), padding="same"),
            conv2D_batchnorm(channel_out // 4, (1, 1), padding="same"),
        ]
    )


def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    return loss


class global_context(layers.Layer):
    def __init__(self, ksize, strides):
        super().__init__()
        self.ksize = ksize
        self.strides = strides

    def call(self, channel_in):
        x = layers.AveragePooling2D(
            pool_size=self.ksize, strides=self.strides, padding="same"
        )(channel_in)

        cx = layers.Lambda(lambda e: tf.math.square(e))(x)
        cx = layers.Lambda(lambda e: tf.math.reduce_mean(e))(cx)

        out = layers.Lambda(lambda e: tf.math.divide(e[0], e[1]))([x, cx])

        return out


def LPRNet(
    n_classes,
    shape=config.INPUT_DIMS,
):
    input_layer = layers.Input(shape)

    x = conv2D_batchnorm(64, (3, 3), strides=1, padding="same")(input_layer)

    x = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    x2 = basic_block(128)(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 2), padding="same")(x2)
    x3 = basic_block(256)(x)
    x = basic_block(256)(x3)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 2), padding="same")(x)
    x = layers.Dropout(0.5)(x)

    x = conv2D_batchnorm(256, (4, 1), strides=1, padding="same")(x)

    x = layers.Dropout(0.5)(x)

    x = conv2D_batchnorm(n_classes, (1, 13), padding="same")(x)

    # Global Context
    cx = layers.Lambda(lambda e: tf.math.square(e))(x)
    cx = layers.Lambda(lambda e: tf.math.reduce_mean(e))(cx)
    x0 = layers.Lambda(lambda e: tf.math.divide(e[0], e[1]))([x, cx])

    x1 = global_context(
        ksize=[1, 4],
        strides=[1, 4],
    )(input_layer)
    x2 = global_context(
        ksize=[1, 4],
        strides=[1, 4],
    )(x2)
    x3 = global_context(
        ksize=[1, 2],
        strides=[1, 2],
    )(x3)

    x = layers.Lambda(lambda e: tf.concat([e[0], e[1], e[2], e[3]], 3))(
        [x0, x1, x2, x3]
    )
    x = layers.Conv2D(
        n_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
    )(x)
    logits = layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(x)
    output_layer = layers.Softmax()(logits)

    return keras.Model(input_layer, output_layer)
