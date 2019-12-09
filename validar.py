from keras.models import load_model
from keras.models import model_from_json
import json

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_saved.h5')

model.