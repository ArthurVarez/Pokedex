import typing as t

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .. import config
from . import resnet
from . import alexnet


def get_model() -> Model:
    raw_model: t.Dict = config.get("models")
    name = raw_model.get("name")

    if name == "alexnet":
        print("\nUsing AlexNet as image classifier.\n")
        return alexnet.AlexNet.build_alexnet()
    if name == "resnet":
        raw_model: t.Dict = config.get("models")
        layers = raw_model["layers"]
        print(f"\nUsing ResNet{layers} as image classifier.\n")
        return resnet.build_resnet()

    raise RuntimeError(f"Unknown models {name}.")


def compile_model(model: Model) -> None:
    loss = config.get("models", "loss")
    optimizer = config.get("models", "optimizer")
    metrics = config.get("models", "metrics")

    learning_rate = config.get("models", "learning_rate")
    momentum = config.get("models", "momentum")
    if optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )


def train(
    model: Model,
    train_dataset: ImageDataGenerator,
    valid_dataset: ImageDataGenerator,
    callbacks: t.List[Callback],
) -> None:
    epochs = config.get("training", "epochs")
    verbose = config.get("training", "verbose", default=1)

    nb_train_samples = train_dataset.samples
    nb_validation_samples = valid_dataset.samples
    train_batch_size = config.get("training", "batch_size")
    validation_batch_size = config.get("validation", "batch_size")

    model.fit(
        train_dataset,
        steps_per_epoch=nb_train_samples // train_batch_size,
        epochs=epochs,
        validation_data=valid_dataset,
        validation_steps=nb_validation_samples // validation_batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
