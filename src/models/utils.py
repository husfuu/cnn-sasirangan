import tensorflow as tf
from keras import Input, Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D, MaxPooling2D, SeparableConv2D)
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_acc') > 0.99:
            print("\nReached accuracy threshold! Terminating training.")
            self.model.stop_training = True

    def on_save_best_model():
        return tf.keras.callbacks.ModelCheckpoint(
            filepath="cnn_model.h5",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )


def construct_model(image_size: list, kernel_size: tuple) -> Sequential:
    """
    """
    model = Sequential()

    # first convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))

    # second convolutional layer
    model.add(Conv2D(128, kernel_size))
    model.add(MaxPooling2D(2, 2))

    # flatten the output from the convolutional layer
    model.add(Flatten())
    model.add(Dropout(0.5))

    return model
