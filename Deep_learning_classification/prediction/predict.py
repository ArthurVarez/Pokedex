import gradio as gr
import os 
import numpy as np
import sys
from tensorflow import load_model
from tensorflow.keras.preprocessing import image
from tensorflow import keras


global model_path
global model
global classes
def inference(data):
    
    img = image.img_to_array(data)
    img /= 255
    img = np.expand_dims(img, axis=0)
    
  
    prediction = model.predict(img)[0]
    round_prediction = [round(i, 5) for i in prediction]
    prediction_index = np.argmax(round_prediction)
    prediction_dict = dict(zip(classes,round_prediction))
    sorted_prediction_dict = dict(sorted(prediction_dict.items(),key = lambda x:x[1],reverse=True))
    print(sorted_prediction_dict)
    res = ""
    for k,v in sorted_prediction_dict.items():
      res+=(f"class: {k} , probability:{v} \n")
    return str(sorted_prediction_dict)



if __name__=="__main__":
	model_path = ""
	model = load_model(model_path)
	classes= ["Arbok", "Aspicot", "Caninos","Chetiflor","Dracolosse","Ectoplasma","Evoli","Hypotrempe","Rondoudou","Ronflex"]
	gr.Interface(fn=inference, 
             inputs = gr.inputs.Image(shape=(128,128)) , 
             outputs="text").launch(share=True,debug=True)









__