import datetime
import os
import argparse
import json
from core.models.ConvRNNRecognizer import ConvRNNRecognitionModel
from core.models.MobileNetDetector import MobileNetV2Detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", help="set output width")
    args = parser.parse_args()
    if args.model:
        if args.model == 'recognition':
            print("training %s model" % args.model)
            with open(os.path.join('configuration', 'recognition.json'), 'r') as f:
                model_conf = json.load(f)
            recognizer = ConvRNNRecognitionModel(model_conf)
            recognizer.MODEL.summary()
            run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
            recognizer.train(run_name, 0, 1)

        elif args.model == 'detection':
            print("training %s model" % args.model)
            with open(os.path.join('configuration', 'detection.json'), 'r') as f:
                model_conf = json.load(f)
            detector = MobileNetV2Detector(model_conf)
            detector.MODEL.summary()
            detector.train()
        else:
            print('wrong model type')
