import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import json
from keras.models import model_from_json

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_saved.h5')


image_path="Validar\ovo.jpg"
img = image.load_img(image_path, target_size=(128, 128))
plt.imshow(img)
img = np.expand_dims(img, axis=0)
result = model.predict_classes(img)
plt.title(get_label_name(result[0][0]))
plt.show()