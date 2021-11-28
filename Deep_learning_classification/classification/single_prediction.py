import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

model = load_model('best_models_mobilenet/best_models_MobileNet_64x64_BS4_noshuffle/models-050-0.986085-0.987739.h5')

img = image.load_img('data/images_test/lying_lateral/video1_frame_000009_pig_7_Lying lateral_.png', target_size=(64, 64))
img = image.img_to_array(img)
img /= 255
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0]
round_prediction = [round(i, 5) for i in prediction]

prediction_index = np.argmax(round_prediction)

print(f'\nPrediction : {round_prediction}\n')

if prediction_index == 0:
    print('\nLying lateral\n')
elif prediction_index == 1:
    print('\nLying sternal\n')
else:
    print('\nStanding\n')

########################################################
# {'Lying lateral': 0, 'Lying sternal': 1, 'Standing': 2}
#########################################################

