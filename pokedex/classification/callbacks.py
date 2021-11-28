import typing as t
import sys
from tensorflow.keras import callbacks as c


if(os.getcwd()=="/content"):
    import config
else:
    from . import config

AVAILABLE_CALLBACKS: t.Dict[str, t.Type[c.Callback]] = {
    "reduce_lr_on_plateau": c.ReduceLROnPlateau,
    "early_stopping": c.EarlyStopping,
    "csv_logger": c.CSVLogger,
    "tensorboard": c.TensorBoard,
    "model_checkpoint": c.ModelCheckpoint,
}


def load() -> t.List[c.Callback]:
    if not config.has("callbacks"):
        return []

    callbacks: t.List[c.Callback] = []
    raw_callbacks: t.Dict[str, t.Dict] = config.get("callbacks")
    for name, arguments in raw_callbacks.items():
        if name not in AVAILABLE_CALLBACKS:
            raise KeyError(f"Unknown callback {name}.")

        callback = _create_callback(name, arguments)
        callbacks.append(callback)

    return callbacks


def _create_callback(name: str, arguments: t.Dict[str, t.Any]) -> c.Callback:
    if name == "csv_logger":
        return _build_csv_logger(arguments)
    if name == "tensorboard":
        return _build_tensorboard(arguments)
    if name == "model_checkpoint":
        return _build_model_checkpoint(arguments)

    return AVAILABLE_CALLBACKS[name](**arguments)


def get_basename() -> str:
    name = config.get("models", "name")
    img_rows = config.get("models", "img_rows")
    img_cols = config.get("models", "img_cols")
    lr = config.get("models", "learning_rate")
    batch_size = config.get("training", "batch_size")

    if name == "resnet":
        layer = config.get("models", "layers")
        return f"{name}{layer}_({img_rows}-{img_cols})_{lr}_{batch_size}"
    else:
        return f"{name}_({img_rows}-{img_cols})_{lr}_{batch_size}"


def _get_fullname() -> str:
    basename = get_basename()
    return f"{basename}"


def _build_csv_logger(arguments: t.Dict[str, t.Any]) -> c.Callback:
    filename = arguments.pop("filename", "auto")
    path = config.get("filepath", "root_folder_path")
    dataset = config.get("models", "dataset")

    if filename == "auto":
        filename = get_basename()
        filename = f"{path}/trainings/{dataset}/{filename}/{filename}.csv"

    return c.CSVLogger(filename, **arguments)


def _build_tensorboard(arguments: t.Dict[str, t.Any]) -> c.Callback:
    log_dir = arguments.pop("log_dir", "auto")
    path = config.get("filepath", "root_folder_path")
    dataset = config.get("models", "dataset")
    filename = get_basename()

    if log_dir == "auto":
        log_dir = _get_fullname()
        log_dir = f"{path}/trainings/{dataset}/{filename}/tensorboard_logs/{log_dir}"

    return c.TensorBoard(log_dir, **arguments)


def _build_model_checkpoint(arguments: t.Dict[str, t.Any]) -> c.Callback:
    basename = get_basename()
    filepath = arguments.pop("filepath", "auto")
    path = config.get("filepath", "root_folder_path")
    dataset = config.get("models", "dataset")
    filename = get_basename()

    if filepath == "auto":
        filepath = f'{path}/trainings/{dataset}/{filename}/models/{basename}' + "_{epoch:03d}-{accuracy:03f}-{" \
                                                                                 "val_accuracy:03f}" + ".h5"

    return c.ModelCheckpoint(filepath, **arguments)
