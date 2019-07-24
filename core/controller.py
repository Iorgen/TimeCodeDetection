import cv2
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from train.recognition_train import decode_predict_ctc
from inference import recognition
from inference import detection


class Controller():

    def __init__(self):
        # TODO load from configuration
        self.graph = tf.get_default_graph()
        self.IMAGE_SIZE = 224
        self.IOU_THRESHOLD = 0.5
        self.SCORE_THRESHOLD = 0.2
        self.MAX_OUTPUT_SIZE = 49
        self.RECOGNITION_WEIGHT_FILE = os.path.join('inference', 'weights', 'recognition',  'weights19.h5')
        self.DETECTION_WEIGHT_FILE = os.path.join('inference', 'weights', 'detection',  'model-0.44.h5')
        self.VIDEO_FOLDER = os.path.join('static', 'video')
        self.IMAGE_FOLDER = os.path.join('static', 'image')
        self.recognizer = recognition.Recognizer(self.RECOGNITION_WEIGHT_FILE)
        self.detector = detection.Detector(self.DETECTION_WEIGHT_FILE)

    def video_recognition(self, video):
        predictions = {}
        count = 0
        video_filename = datetime.today().strftime('%Y-%m-%d') + video.filename
        video_file_path = os.path.join(self.VIDEO_FOLDER, video_filename)
        print(video_filename)
        video.save(video_file_path)
        del video
        vidcap = cv2.VideoCapture(video_file_path)
        vidcap.set(cv2.CAP_PROP_FPS, 1)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, 10000)
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 + 15000))  # added this line
            success, image = vidcap.read()
            time_code_crops = self.image_detection(image)

            for time_code in time_code_crops:
                crop_file_name = video_filename + str(count) + '.jpg'
                cv2.imwrite(os.path.join(self.IMAGE_FOLDER, crop_file_name), time_code)
                predictions[crop_file_name] = self.crop_recognition(time_code)

            if count == 3:
                success = False
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
            print(pred)
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
                print(h, w)
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
        img = cv2.resize(img, (135, 35))
        img = img.reshape(1, 35, 135)
        expand_img = np.expand_dims(img.T, axis=0)
        with self.graph.as_default():
            self.recognizer.load_weights(self.RECOGNITION_WEIGHT_FILE)
            net_out_value = self.recognizer.predict(expand_img)
            pred_texts = decode_predict_ctc(net_out_value)
        return str(pred_texts)



