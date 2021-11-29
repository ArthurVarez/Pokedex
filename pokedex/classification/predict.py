import gradio as gr
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path

global model_path
global models
global classes


def load_models():
    models_ = dict()
    current_directory = Path.cwd()
    parent_directory = current_directory.parent.absolute()
    models_path = str(Path.joinpath(parent_directory, "best_models"))
    if os.path.isdir(models_path) is False:
        return
    for filename in os.listdir(models_path):
        path_ = models_path + '/' + filename
        if filename.endswith(".h5"):
            models_[filename] = load_model(path_)
    global models
    models = models_


def inference(data):
    res = ""
    for name, model in models.items():
        # dim = list(name.split("_"))[1][1:-1]
        # x, y = dim.split("-")
        img = image.img_to_array(data)
        img /= 255
        img = np.expand_dims(img, axis=0)
        res += f"Model : {name[:-3]}\n\n"
        predictions = model.predict(img)[0]
        round_prediction = [i for i in predictions]
        prediction_dict = dict(zip(classes, round_prediction))
        sorted_prediction_dict = dict(sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True))
        keys = list(sorted_prediction_dict.keys())[:2]
        for elem in keys:
            probability = float(sorted_prediction_dict[elem] * 100)
            res += str(f"\t{elem} : {round(probability, 2)}%\n")
        res += "\n"
    return res


if __name__ == "__main__":
    load_models()
    classes = ["Arbok", "Aspicot", "Caninos", "Chetiflor", "Dracolosse", "Ectoplasma", "Evoli", "Hypotrempe",
               "Rondoudou", "Ronflex"]

    gr.Interface(fn=inference,
                 inputs=gr.inputs.Image(shape=(64, 64)),
                 outputs="text").launch(share=True, debug=True)
