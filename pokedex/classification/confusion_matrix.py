import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path

dataset = "10_pokemons"
current_directory = Path.cwd()
parent_directory = current_directory.parent.absolute()
test_data_path = Path.joinpath(parent_directory, f"data/{dataset}/validation")
chosen_model = "resnet18_(64-64)_0.001_16_042-0.918018-0.901316"
model_path = Path.joinpath(parent_directory, f"best_models/{chosen_model}.h5")

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(64, 64),
                                                        batch_size=1,
                                                        shuffle=False,
                                                        class_mode='categorical')

model = tf.keras.models.load_model(model_path)

Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

labels = ["Arbok", "Aspicot", "Caninos", "Chetiflor", "Dracolosse", "Ectoplasma", "Evoli", "Hypotrempe",
          "Rondoudou", "Ronflex"]

cm = confusion_matrix(validation_generator.classes, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

TruePositive = np.diag(cm)
FalsePositive = []
for i in range(len(labels)):
    FalsePositive.append(sum(cm[:, i]) - cm[i, i])
FalseNegative = []
for i in range(len(labels)):
    FalseNegative.append(sum(cm[i, :]) - cm[i, i])
    TrueNegative = []
for i in range(len(labels)):
    temp = np.delete(cm, i, 0)  # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TrueNegative.append(sum(sum(temp)))

print(TruePositive)
print(FalsePositive)
print(FalseNegative)
print(TrueNegative)

data = {"TruePositive": TruePositive, "FalsePositive": FalsePositive, "FalseNegative": FalseNegative,
        "TrueNegative": TrueNegative}
metrics = pd.DataFrame(index=labels, data=data)
metrics["TVP"] = metrics.TruePositive / (metrics.TruePositive + metrics.FalseNegative)
metrics["TFP"] = metrics.FalsePositive / (metrics.FalsePositive + metrics.TrueNegative)
metrics["Precision"] = metrics.TruePositive / (metrics.TruePositive + metrics.FalsePositive)
kappa = sum(TruePositive) / validation_generator.samples

try:
    metrics.to_csv(f"{parent_directory}/best_models/eval/{chosen_model}_metrics_kappa={kappa}")
except FileNotFoundError:
    Path.mkdir(Path.joinpath(parent_directory, "best_models/eval"))
    metrics.to_csv(f"{parent_directory}/best_models/eval/{chosen_model}_metrics_kappa={kappa}")

# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
#
# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # False discovery rate
# FDR = FP/(TP+FP)
#
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
#
# print(FP)
# print(FN)
# print(TP)
# print(TN)
# print(f"Recall : {TPR}")
# print(f"Precision : {PPV}")
