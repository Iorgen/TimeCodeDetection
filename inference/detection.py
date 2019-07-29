from core.detection_model import init_detection_model


def detector(model_conf):
    model = init_detection_model(model_conf)
    return model
