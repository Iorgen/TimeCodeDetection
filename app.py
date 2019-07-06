from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
from inference import recognition
from train.augmentation import decode_predict_ctc
import numpy as np
import requests
import sys, os
import cv2
import json
import tensorflow as tf
from keras import backend as K

VIDEO_FOLDER = os.path.join('static', 'video')
IMAGE_FOLDER = os.path.join('static', 'image')
WEIGHT_FILE = os.path.join('inference', 'weights', 'weights19.h5')
app = Flask(__name__)
api = Api(app)
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['WEIGHT_FILE'] = WEIGHT_FILE
recognizer = recognition.Recognizer(app.config['WEIGHT_FILE'])
global graph
graph = tf.get_default_graph()


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def recognition():
    # tf.keras.backend.clear_session()
    # K.clear_session()
    result = dict()
    count = 0
    video = request.files['video']
    video_file_path = os.path.join(app.config['VIDEO_FOLDER'], video.filename)
    video.save(video_file_path)
    vidcap = cv2.VideoCapture(video_file_path)
    vidcap.set(cv2.CAP_PROP_FPS, 1)
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # TODO log saver
        # print('Read a new frame: ', success)
        img_file_name = "_frame%d.jpg" % count
        print(img_file_name)
        cv2.imwrite(os.path.join(app.config['IMAGE_FOLDER'], img_file_name), image)
        # ---------------------------------
        # TODO time code detection from the article
        # Here only cap at this moment
        # TODO fix some bugs with png
        test_img_path = os.path.join(app.config['IMAGE_FOLDER'], 'nws.jpg')
        img = cv2.imread(test_img_path, 0)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (135, 35))
        img = img.reshape(1, 35, 135)
        expand_img = np.expand_dims(img.T, axis=0)
        pred_texts = ""
        with graph.as_default():
            net_out_value = recognizer.predict(expand_img)
            pred_texts = decode_predict_ctc(net_out_value)

        print(pred_texts)
        # timecodeClean = {
        #     'hours': pred_texts[0],
        #     'minutes': pred_texts[1],
        #     'seconds': pred_texts[2]
        # }
        success = False

    # TODO splitting function
    # TODO analyser that get early and later values and push them as output
    # TODO save time from and to indto video description (About)

    return render_template("index.html", timecode=pred_texts,  init=True)


if __name__ == '__main__':
    app.run(debug=True)
