import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay

test_data_path = "F:/Programmation/UQAC/data_mining/projet/pokedex/data/10_pokemons/validation"

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(128, 128),
                                                        batch_size=1,
                                                        shuffle=False,
                                                        class_mode='categorical')

model = tf.keras.models.load_model("./alexnet.h5")

Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

labels = ["Arbok", "Aspicot", "Caninos", "Chetiflor", "Dracolosse", "Ectoplasma", "Evoli", "Hypotrempe",
          "Rondoudou", "Ronflex"]

print(Y_pred)
cm = confusion_matrix(validation_generator.classes, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

TruePositive = np.diag(cm)
FalsePositive = []
for i in range(len(labels)):
    FalsePositive.append(sum(cm[:,i]) - cm[i,i])
FalseNegative = []
for i in range(len(labels)):
    FalseNegative.append(sum(cm[i,:]) - cm[i,i])
    TrueNegative = []
for i in range(len(labels)):
    temp = np.delete(cm, i, 0)   # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TrueNegative.append(sum(sum(temp)))
    
print(TruePositive)
print(FalsePositive)
print(FalseNegative)
print(TrueNegative)


data = {"TruePositive":TruePositive,"FalsePositive":FalsePositive,"FalseNegative":FalseNegative,
        "TrueNegative":TrueNegative}
metrics = pd.DataFrame(index = labels,data=data)
metrics["TVP"] = metrics.TruePositive/(metrics.TruePositive+metrics.FalseNegative)
metrics["TFP"] = metrics.FalsePositive/(metrics.FalsePositive+metrics.TrueNegative)
metrics["Precision"] = metrics.TruePositive/(metrics.TruePositive+metrics.FalsePositive)
kappa = sum(TruePositive)/validation_generator.samples

metrics.to_csv(f"./{model[2:]}_metrics_kappa={kappa}")






