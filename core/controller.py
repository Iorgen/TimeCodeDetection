import cv2
import os
import json
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from train.recognition_train import decode_predict_ctc
from inference import recognition
from inference import detection
from core.singleton import Singleton


class TimeCodeController(metaclass=Singleton):

    def __init__(self):
        with open('configuration/recognition.json', 'r') as f:
            self.recognition_model_conf = json.load(f)

        with open('configuration/detection.json', 'r') as f:
            self.detection_model_conf = json.load(f)

        self.graph = tf.get_default_graph()
        self.IMAGE_SIZE = self.detection_model_conf["IMAGE_SIZE"]
        self.IOU_THRESHOLD = self.detection_model_conf["IOU_THRESHOLD"]
        self.SCORE_THRESHOLD = self.detection_model_conf["SCORE_THRESHOLD"]
        self.MAX_OUTPUT_SIZE = self.detection_model_conf["MAX_OUTPUT_SIZE"]
        self.RECOGNITION_WEIGHT_FILE = os.path.join('inference', 'weights', 'recognition',
                                                    self.recognition_model_conf["WEIGHTS"])
        self.DETECTION_WEIGHT_FILE = os.path.join('inference', 'weights', 'detection',
                                                  self.detection_model_conf["WEIGHTS"])
        self.VIDEO_FOLDER = os.path.join('app', 'static', 'video')
        self.IMAGE_FOLDER = os.path.join('app', 'static', 'image')
        self.recognizer = recognition.recognizer(self.recognition_model_conf)
        self.detector = detection.detector(self.detection_model_conf)

    def video_recognition(self, video):
        predictions = {}
        count = 0
        video_filename = datetime.today().strftime('%Y-%m-%d') + video.filename
        video_file_path = os.path.join(self.VIDEO_FOLDER, video_filename)
        video.save(video_file_path)
        del video
        vidcap = cv2.VideoCapture(video_file_path)
        vidcap.set(cv2.CAP_PROP_FPS, 1)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, 10000)
        success = True
        while success:
            try:
                vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 + 20000))  # added this line
                success, image = vidcap.read()
                cv2.imwrite(os.path.join(self.IMAGE_FOLDER, video_filename + str(count) + '.jpg'), image)
                time_code_crops = self.image_detection(image)

                for time_code in time_code_crops:
                    crop_file_name = video_filename + str(count) + '.jpg'
                    cv2.imwrite(os.path.join(self.IMAGE_FOLDER, crop_file_name), time_code)
                    predictions[crop_file_name] = self.crop_recognition(time_code)
            except Exception as e:
                print(e)
                success = False
            # if count == 3:
            #     success = False
            count += 1

        # TODO analyser that get early and later values and push them as output
        # TODO save time from and to indto video description (About)
        # TODO windows bug deleting the file
        os.remove(video_file_path)
        return predictions

    def image_detection(self, image):
        with self.graph.as_default():
            crops = list()
            unscaled = image
            img = cv2.resize(unscaled, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            feat_scaled = preprocess_input(np.array(img, dtype=np.float32))
            self.detector.load_weights(self.DETECTION_WEIGHT_FILE)
            pred = np.squeeze(self.detector.predict(feat_scaled[np.newaxis, :]))
            height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]
            coords = np.arange(pred.shape[0] * pred.shape[1])
            y = (y_f + coords // pred.shape[0]) / (pred.shape[0] - 1)
            x = (x_f + coords % pred.shape[1]) / (pred.shape[1] - 1)
            boxes = np.stack([y, x, height, width, score], axis=-1)
            boxes = boxes[np.where(boxes[..., -1] >= self.SCORE_THRESHOLD)]
            selected_indices = tf.image.non_max_suppression(boxes[..., :-1], boxes[..., -1],
                                                            self.MAX_OUTPUT_SIZE,
                                                            self.IOU_THRESHOLD)
            selected_indices = tf.Session().run(selected_indices)
            for y_c, x_c, h, w, _ in boxes[selected_indices]:
                x0 = unscaled.shape[1] * (x_c - w / 2)
                y0 = unscaled.shape[0] * (y_c - h / 2)
                x1 = x0 + unscaled.shape[1] * w
                y1 = y0 + unscaled.shape[0] * h
                crop_img = unscaled[int(y0):int(y1), int(x0):int(x1)]
                crops.append(crop_img)
                # cv2.imwrite("detected-%.3d.jpg" % randint(1, 200), crop_img)
        return crops

    def crop_recognition(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (135, 35))
        img = img.reshape(1, 35, 135)
        expand_img = np.expand_dims(img.T, axis=0)
        with self.graph.as_default():
            self.recognizer.load_weights(self.RECOGNITION_WEIGHT_FILE)
            net_out_value = self.recognizer.predict(expand_img)
            pred_texts = decode_predict_ctc(net_out_value)
        return str(pred_texts)



