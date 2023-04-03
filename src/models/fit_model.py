import tensorflow as tf
from keras import Input, Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D, SeparableConv2D)
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.utils.vis_utils import plot_model

from src.models.utils import MyCallback, construct_model


def fit_model() -> None:
    model = construct_model_v2(image_size=IMAGE_SIZE, act='relu')
    my_callback = MyCallback()

    CALLBACKS = [my_callback]

    model.compile(optimizer='adam',
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=METRICS)

    EPOCHS = 200

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        callbacks=CALLBACKS,
                        epochs=EPOCHS)
