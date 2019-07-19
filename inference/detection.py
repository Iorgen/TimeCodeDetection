from core.detection_model import init_detection_model
import json


def Detector(weight_file):
    with open('configuration/recognition.json', 'r') as f:
        model_conf = json.load(f)

    model = init_detection_model()
    # model.load_weights(weight_file)
    return model
