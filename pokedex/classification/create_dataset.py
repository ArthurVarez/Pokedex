import os

from shutil import copyfile
from pathlib import Path

current_directory = Path.cwd()
root_dir = str(current_directory.parent.absolute())
dataset_name = "10_pokemons"


def _move_to_dir(path: str):
    try:
        os.chdir(path)
    except FileNotFoundError:
        os.mkdir(path)
        os.chdir(path)


def _duplicate(directory: str, pokemon: str, subdir: str, file) -> None:
    _move_to_dir(f"{root_dir}/data/{dataset_name}")
    _move_to_dir(f"{root_dir}/data/{dataset_name}/{directory}/")
    _move_to_dir(f"{root_dir}/data/{dataset_name}/{directory}/" + pokemon)

    copyfile(root_dir + f"/{dataset_name}/{subdir}/{file}",
             root_dir + f"/data/{dataset_name}/{directory}/{subdir}/{file}")


def _sort(ratio: float) -> None:
    for subdir, dirs, files in os.walk(f"{root_dir}/{dataset_name}"):
        count = 0
        for file in files:
            count += 1
            if count / len(files) < ratio:
                _duplicate("training", os.path.basename(subdir), os.path.basename(subdir), file)
            else:
                _duplicate("validation", os.path.basename(subdir), os.path.basename(subdir), file)


_sort(ratio=0.8)
