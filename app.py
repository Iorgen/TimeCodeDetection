import numpy as np
import os
import cv2
import tensorflow as tf
from random import randint
from matplotlib import pyplot
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_restful import reqparse, abort, Api
from inference import recognition
from inference import detection
from train.recognition_train import decode_predict_ctc
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
global graph
graph = tf.get_default_graph()

# TODO ALL THAT SHIT INTO CONFIGURATION FILES
IMAGE_SIZE = 224
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.2
MAX_OUTPUT_SIZE = 49

VIDEO_FOLDER = os.path.join('static', 'video')
IMAGE_FOLDER = os.path.join('static', 'image')
RECOGNITION_WEIGHT_FILE = os.path.join('inference', 'weights', 'recognition',  'weights19.h5')
DETECTION_WEIGHT_FILE = os.path.join('inference', 'weights', 'detection',  'model-0.18.h5')
app = Flask(__name__)
api = Api(app)
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['RECOGNITION_WEIGHT_FILE'] = RECOGNITION_WEIGHT_FILE
app.config['DETECTION_WEIGHT_FILE'] = DETECTION_WEIGHT_FILE
recognizer = recognition.Recognizer(app.config['RECOGNITION_WEIGHT_FILE'])
detector = detection.Detector(app.config['DETECTION_WEIGHT_FILE'])


tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})


@app.route('/', methods=['POST', 'GET'])
def recognition():
    pred_texts = list()
    count = 0
    # TODO No video error
    video = request.files['video']
    video_filename = datetime.today().strftime('%Y-%m-%d') + video.filename
    video_file_path = os.path.join(app.config['VIDEO_FOLDER'], video_filename)
    print(video_filename)
    video.save(video_file_path)
    del video
    vidcap = cv2.VideoCapture(video_file_path)
    vidcap.set(cv2.CAP_PROP_FPS, 1)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, 10000)
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 + 10000))  # added this line
        success, image = vidcap.read()
        time_code_crops = det_prediction(image)

        for time_code in time_code_crops:
            pred_texts.append(rec_prediction(time_code))

        # --------------------------------------------------------
        # TODO fix some bugs with png
        # pred_texts = rec_prediction(filename='some2.jpg')
        # print(pred_texts)
        # try:
        #     # TODO write in log file all that stuff
        #     pred_texts = pred_texts[0].split(':')
        # except Exception as e:
        #     # TODO write in log file all that stuff
        #     print(pred_texts, e)

        if count == 3:
            success = False
        count +=1

    # TODO analyser that get early and later values and push them as output

    # TODO save time from and to indto video description (About)
    os.remove(video_file_path)
    return render_template("index.html", timecode=pred_texts,  init=True)


def det_prediction(image):
    with graph.as_default():
        crops = list()
        unscaled = image
        img = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
        feat_scaled = preprocess_input(np.array(img, dtype=np.float32))
        detector.load_weights(app.config['DETECTION_WEIGHT_FILE'])
        pred = np.squeeze(detector.predict(feat_scaled[np.newaxis, :]))
        print(pred)
        height, width, y_f, x_f, score = [a.flatten() for a in np.split(pred, pred.shape[-1], axis=-1)]
        coords = np.arange(pred.shape[0] * pred.shape[1])
        y = (y_f + coords // pred.shape[0]) / (pred.shape[0] - 1)
        x = (x_f + coords % pred.shape[1]) / (pred.shape[1] - 1)
        boxes = np.stack([y, x, height, width, score], axis=-1)
        boxes = boxes[np.where(boxes[..., -1] >= SCORE_THRESHOLD)]
        selected_indices = tf.image.non_max_suppression(boxes[..., :-1], boxes[..., -1], MAX_OUTPUT_SIZE, IOU_THRESHOLD)
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


# TODO change input signature on image type
def rec_prediction(image):
    # test_img_path = os.path.join(app.config['IMAGE_FOLDER'], image)
    # img = cv2.imread(test_img_path, 0)
    # pyplot.imshow(image)
    # pyplot.show()
    # img = cv2.bitwise_not(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pyplot.imshow(img)
    # pyplot.show()
    img = cv2.resize(img, (135, 35))
    img = img.reshape(1, 35, 135)
    expand_img = np.expand_dims(img.T, axis=0)
    with graph.as_default():
        recognizer.load_weights(app.config['RECOGNITION_WEIGHT_FILE'])
        net_out_value = recognizer.predict(expand_img)
        pred_texts = decode_predict_ctc(net_out_value)
    return pred_texts


if __name__ == '__main__':
    app.run(debug=True)
