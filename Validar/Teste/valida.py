import tensorflow.keras
from PIL import Image
import numpy as np

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open(r'15.jpeg')
image = image.resize((224, 224))
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)
