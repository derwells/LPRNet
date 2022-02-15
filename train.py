import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
import constants
from model.lprnet import LPRNet
from classes.data_generator import DataGenerator


def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    return loss


def build_model():
    model = LPRNet(constants.N_OUTPUTS)
    learning_rate_scheduler = keras.optimizers.schedules.ExponentialDecay(
        1e-3, 
        decay_steps=500,
        decay_rate=0.995,
        staircase=True
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate_scheduler
        ),
        loss=ctc_loss
    )

    return model


if __name__ == "__main__":
    model = build_model()

    train_data = config.CLEAN_SPLIT_PATHS["train"]
    training_generator = DataGenerator(
        os.listdir(train_data),
        train_data
    )

    validation_data = config.CLEAN_SPLIT_PATHS["val"]
    validation_generator = DataGenerator(
        os.listdir(validation_data),
        validation_data
    )

    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10
        ),
    ]

    # Train model
    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Save as .tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(config.MODEL_TARGET_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"Done training {config.MODEL_TARGET_PATH}")
