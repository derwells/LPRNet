import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

import constants
import config
from classes.license_plate import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,
        fnames,
        fdir,
        batch_size=32,
        dim=config.INPUT_DIMS,
        shuffle=True,
    ): 
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.fnames = fnames
        self.fdir = fdir
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.fnames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        fnames_temp = [self.fnames[idx] for idx in indices]

        # Generate data
        X, y = self.__data_generation(fnames_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.fnames))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, fnames_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=np.ndarray)

        # Generate data
        for i, fname in enumerate(fnames_temp):
            # Store sample
            img = cv2.imread(
                os.path.join(
                    self.fdir, fname
                )
            )
            img = cv2.resize(img, (94, 24))/256
            X[i,] = img

            # Store class
            license_plate = CroppedLicensePlate(fname)
            y[i] = license_plate.lpn

        return (
            tf.convert_to_tensor(X),
            tf.ragged.constant(y).to_tensor()
        )
