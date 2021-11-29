import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def LoadTensorboards():
    df_tensorboards = dict()
    current_directory = Path.cwd()
    parent_directory = current_directory.parent.absolute()
    metrics_path = str(Path.joinpath(parent_directory, "metrics"))

    for csv in os.listdir(metrics_path):
        if csv.endswith("csv"):
            df_tensorboards[str(csv)] = (pd.read_csv(os.path.join(metrics_path, csv)))
    return df_tensorboards


def showmcuvemodel(title, df):
    key_split = list(title.split("_"))
    fig, axes = plt.subplots(figsize=(15.0, 15.0), nrows=2, ncols=1)
    title_ = f"model = {key_split[0]}, image size = {key_split[1]}, learning rate = {key_split[2]}"

    axes[0].plot(df.epoch, df.val_loss, label="validation loss")
    axes[0].plot(df.epoch, df.loss, label="training loss")
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('train set metric (validation loss vs training loss)')
    axes[0].legend()

    fig.suptitle(title_, fontsize=16)

    axes[1].plot(df.epoch, df.val_accuracy, label="validation accuracy")
    axes[1].plot(df.epoch, df.accuracy, label="training accuracy")
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('test set metric (validation accuracy vs training accuracy)')
    axes[1].legend()

    current_directory = Path.cwd()
    parent_directory = current_directory.parent.absolute()
    metricsviz_path = str(Path.joinpath(parent_directory, "metrics/metricsviz"))

    folderexist(metricsviz_path)
    plt.savefig(f"{metricsviz_path}/{title}.png")


def showmetrics(df, metric1, metric2):
    list_metric1 = []
    list_metric2 = []
    index = list(df.keys())
    for k, v in df.items():
        list_metric1.append(v[metric1].max())
        list_metric2.append(v[metric2].min())

    df = pd.DataFrame({metric1: list_metric1,
                       metric2: list_metric2}, index=index)
    ax = df.plot.bar(rot=45, color={metric1: "green", metric2: "red"})

    current_directory = Path.cwd()
    parent_directory = current_directory.parent.absolute()
    metricsviz_path = str(Path.joinpath(parent_directory, "metrics/metricsviz"))

    folderexist(metricsviz_path)
    plt.savefig(f"{metricsviz_path}/{metric1},{metric2},{index}.png")


def folderexist(newpath: str):
    try:
        # Create target Directory
        os.mkdir(newpath)
        print("Directory ", newpath, " Created ")
    except FileExistsError:
        print("Directory ", newpath, " already exists")


if __name__ == "__main__":
    df = LoadTensorboards()
    for k, v in df.items():
        showmcuvemodel(k, v)

    showmetrics(df, "accuracy", "loss")
    showmetrics(df, "val_accuracy", "val_loss")
