import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
