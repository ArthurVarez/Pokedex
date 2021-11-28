import typing as t
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from . import config

_Dataset = t.Tuple[np.ndarray, np.ndarray]

DATAGEN: t.Optional[ImageDataGenerator] = None


def _use_data_augmentation(name: str):
    global DATAGEN
    arguments: t.Dict[str, t.Any] = config.get(name, "data_augmentation")
    DATAGEN = ImageDataGenerator(**arguments)


def create_dataset(name: str, **kwargs) -> t.Union[tf.data.Dataset, ImageDataGenerator]:
    img_rows = config.get("models", "img_rows")
    img_cols = config.get("models", "img_cols")
    class_mode = config.get("models", "class_mode")
    batch_size = config.get(name, "batch_size")
    shuffle = config.get(name, "shuffle", default=False)

    root_path = config.get("filepath", "root_folder_path")
    used_dataset = config.get("models", "dataset")

    if name == 'training':
        if config.has(name, "data_augmentation"):
            print("\nDATA AUGMENTATION ON TRAINING IMAGES\n")
            generator = ImageDataGenerator(
                rescale=1. / 255,
                zoom_range=0.2,
                rotation_range=180,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )
        else:
            generator = ImageDataGenerator(
                rescale=1. / 255
            )
    else:
        generator = ImageDataGenerator(
            rescale=1. / 255
        )

    dataset = generator.flow_from_directory(
        f'{root_path}/data/{used_dataset}/{name}',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        shuffle=shuffle,
        class_mode=class_mode,
    )
    return dataset
