from core.models.MobileNetDetector import MobileNetV2Detector


if __name__ == "__main__":
    detector = MobileNetV2Detector()
    detector.MODEL.summary()
    detector.train()
    # detector.save_model_scheme()
