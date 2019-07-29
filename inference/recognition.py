from keras.models import Model
from core.recognition_model import init_recognition_model


def recognizer(model_conf):
    model, input_data, y_pred = init_recognition_model(model_conf)
    model = Model(inputs=input_data, outputs=y_pred)
    return model

