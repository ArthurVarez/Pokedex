import os
import click
import typing as t
from pathlib import Path
from shutil import copyfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import callbacks, config, models
from .preprocessing import create_dataset

from .callbacks import get_basename


def _save_config(config_file_path: str) -> None:
    name = get_basename()
    path = config.get("filepath", "root_folder_path")
    dataset = config.get("models", "dataset")

    try:
        os.chdir(f'{path}')
        try:
            os.chdir('trainings')
        except:
            os.mkdir('trainings')
            os.chdir('trainings')
            print(f'Folder training/ does not exist : created one')
        try:
            os.chdir(f'{dataset}')
        except:
            os.mkdir(f'{dataset}')
            os.chdir(f'{dataset}')
            print(f'Folder {dataset}/ does not exist : created one')

        if os.path.isdir(f'{name}'):
            exit("This training already exists. Try changing the parameters or renaming/deleting your old "
                 "training.")
        else:
            os.mkdir(f'{name}')

        os.mkdir(f'{name}')
        os.chdir(f'{name}')
        os.mkdir('tensorboard_logs')
        os.mkdir('models')
    except:
        print("")

    config_file = f'{config_file_path}'
    duplicate = f'{path}/trainings/{dataset}/{name}/config.yaml'
    copyfile(config_file, duplicate)


def _configure(config_file: str) -> None:
    config_path = Path(config_file)
    config.load(config_path)


def _prepare_data() -> t.Tuple[ImageDataGenerator, ImageDataGenerator]:
    train_dataset = create_dataset("training")
    valid_dataset = create_dataset("validation")
    return train_dataset, valid_dataset


def _train(train_ds: ImageDataGenerator, valid_ds: ImageDataGenerator) -> None:
    model_callbacks = callbacks.load()
    model = models.get()
    models.build(model)
    models.train(model, train_ds, valid_ds, model_callbacks)


@click.command()
@click.option("--config_file", "-cf", "config_file", type=str)
def main(config_file: str) -> None:
    _configure(config_file)
    _save_config(config_file)
    train_ds, valid_ds = _prepare_data()
    _train(train_ds, valid_ds)


if __name__ == '__main__':
    main()
