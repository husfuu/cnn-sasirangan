import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
import os
from keras.preprocessing.image import ImageDataGenerator

def augmented_data(df: pd.DataFrame, img_size, batch_size):
    datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    generator = datagen.flow_from_dataframe(
        df,
        x_col='file_path',
        y_col='category',
        target_size=img_size,
        class_mode='categorical',
        batch_size=(batch_size)
    )

    return generator


def generate_augmented_data(img_class, data_dir, augmented_dir, size) -> None:
    """
    :param img_class:
    :param data_dir:
    :param augmented_dir:
    :param size:
    :return:
    """
    dataset = []
    # image_classes = os.listdir(data_dir)
    for image in os.listdir(os.path.join(data_dir, img_class)):
        image = io.imread(data_dir+img_class+"/"+image)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((size, size))
        dataset.append(np.array(image))

    x = np.array(dataset)

    datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    i = 0
    for batch in datagen.flow(x,
                              batch_size=16,
                              save_to_dir=augmented_dir + img_class,
                              save_prefix=img_class):
        i += 1
        if i > 10:
            print(i)
            break


