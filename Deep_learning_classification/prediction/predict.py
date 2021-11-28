import gradio as gr
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

global model_path
global models
global classes


def load_models():
    models_ = dict()
    path = os.path.join(os.getcwd(), "models")
    if os.path.isdir(path) is False:
        return
    for filename in os.listdir(path):
        path_ = path + '/' + filename
        if filename.endswith(".h5"):
            models_[filename] = load_model(path_)
    global models
    models = models_


def inference(data):
    img = image.img_to_array(data)
    img /= 255
    img = np.expand_dims(img, axis=0)
    res = ""
    for name, model in models.items():
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
                 inputs=gr.inputs.Image(shape=(128, 128)),
                 outputs="text").launch(share=True, debug=True)
