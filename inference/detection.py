from core.detection_model import init_detection_model
import json


def Detector(weight_file):
    with open('configuration/detection.json', 'r') as f:
        model_conf = json.load(f)

    model = init_detection_model(model_conf)
    return model
