import gradio as gr
import os 
import numpy as np
import sys
from tensorflow import load_model
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import zipfile

global model_path
global models
global classes

def load_models():
  models_ = list()
  path = os.path.join(os.getcwd(),"models")
  for filename in os.listdir(path):
    path_ = path+'/'+filename
    if filename.endswith(".h5"):
      models_.append(load_model(path_))
    if filename.endswith(".zip") | filename.endswith(".rar"):
      try:
        with zipfile.ZipFile(path_, 'r') as zip_ref:
          zip_ref.extractall(path_)

        
      except Exception as e:
        raise
  models = models_

def inference(data):
    
    img = image.img_to_array(data)
    img /= 255
    img = np.expand_dims(img, axis=0)
    res = ""
    for model in models:
      predictions = model.predict(img)[0]
      round_prediction = [round(i, 5) for i in prediction]
      prediction_index = np.argmax(round_prediction)
      prediction_dict = dict(zip(classes,round_prediction))
      sorted_prediction_dict = dict(sorted(prediction_dict.items(),key = lambda x:x[1],reverse=True))
      _ = list(sorted_prediction_dict.keys())[0]
      res+=str(f"{res},{sorted_prediction_dict[res]}\n")
    return res



if __name__=="__main__":
	model_path = ""
	model = load_model(model_path)
	classes= ["Arbok", "Aspicot", "Caninos","Chetiflor","Dracolosse","Ectoplasma","Evoli","Hypotrempe","Rondoudou","Ronflex"]
	gr.Interface(fn=inference, 
             inputs = gr.inputs.Image(shape=(128,128)) , 
             outputs="text").launch(share=True,debug=True)









__