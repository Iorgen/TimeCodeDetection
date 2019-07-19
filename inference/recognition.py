from keras.models import Model
from core.recognition_model import init_recognition_model
import json


def Recognizer(weight_file):
    with open('configuration/recognition.json', 'r') as f:
        model_conf = json.load(f)

    model, input_data, y_pred = init_recognition_model(model_conf)
    # model.load_weights(weight_file)
    model = Model(inputs=input_data, outputs=y_pred)
    return model

