# TODO create individual requirements file for gpu support
import json
import csv
import sys
import math
import os
import numpy as np
import tensorflow as tf
# - For windows path bugs
sys.path.append(sys.path[0] + "/..")
from keras.layers import Conv2D
import pydot as pdt
from keras.utils import plot_model
from core.loss_func import detection_loss
from PIL import Image, ImageDraw, ImageEnhance
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import epsilon


class DataGenerator(Sequence):

    def __init__(self, csv_file, rnd_rescale=False, rnd_multiply=False, rnd_color=False,
                 rnd_crop=False, rnd_flip=False, debug=False, batch_size=32, image_size=448, grid_size=14):
        self.boxes = []
        self.rnd_rescale = rnd_rescale
        self.rnd_multiply = rnd_multiply
        self.rnd_color = rnd_color
        self.rnd_crop = rnd_crop
        self.rnd_flip = rnd_flip
        self.debug = debug
        self.batch_size = batch_size
        self.image_size = image_size
        self.grid_size = grid_size

        with open(csv_file, "r") as file:
            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)
                # x0, y0  top-left corner and x1,y1 bottom-right corner
                path, image_height, image_width, x0, y0, x1, y1, _, _ = row
                self.boxes.append((path, x0, y0, x1, y1))

    def __len__(self):
        return math.ceil(len(self.boxes) / self.batch_size)

    def __getitem__(self, idx):
        boxes = self.boxes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = np.zeros((len(boxes), self.image_size, self.image_size, 3), dtype=np.float32)
        batch_boxes = np.zeros((len(boxes), self.grid_size, self.grid_size, 5), dtype=np.float32)
        for i, row in enumerate(boxes):
            path, x0, y0, x1, y1 = row

            with Image.open(path) as img:
                if self.rnd_rescale:
                    old_width = img.width
                    old_height = img.height

                    rescale = np.random.uniform(low=0.6, high=1.4)
                    new_width = int(old_width * rescale)
                    new_height = int(old_height * rescale)

                    img = img.resize((new_width, new_height))

                    x0 *= new_width / old_width
                    y0 *= new_height / old_height
                    x1 *= new_width / old_width
                    y1 *= new_height / old_height

                if self.rnd_crop:
                    start_x = np.random.randint(0, high=np.floor(0.15 * img.width))
                    stop_x = img.width - np.random.randint(0, high=np.floor(0.15 * img.width))
                    start_y = np.random.randint(0, high=np.floor(0.15 * img.height))
                    stop_y = img.height - np.random.randint(0, high=np.floor(0.15 * img.height))

                    img = img.crop((start_x, start_y, stop_x, stop_y))

                    x0 = max(x0 - start_x, 0)
                    y0 = max(y0 - start_y, 0)
                    x1 = min(x1 - start_x, img.width)
                    y1 = min(y1 - start_y, img.height)

                    # if np.abs(x1 - x0) < 5 or np.abs(y1 - y0) < 5:
                    #     print("\nWarning: cropped too much (obj width {}, obj height {}, img width {}, img height {})\n".format(x1 - x0, y1 - y0, img.width, img.height))

                if self.rnd_flip:
                    elem = np.random.choice([0, 90, 180, 270, 1423, 1234])
                    if elem % 10 == 0:
                        x = x0 - img.width / 2
                        y = y0 - img.height / 2

                        x0 = img.width / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
                        y0 = img.height / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

                        x = x1 - img.width / 2
                        y = y1 - img.height / 2

                        x1 = img.width / 2 + x * np.cos(np.deg2rad(elem)) - y * np.sin(np.deg2rad(elem))
                        y1 = img.height / 2 + x * np.sin(np.deg2rad(elem)) + y * np.cos(np.deg2rad(elem))

                        img = img.rotate(-elem)
                    else:
                        if elem == 1423:
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            y0 = img.height - y0
                            y1 = img.height - y1

                        elif elem == 1234:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            x0 = img.width - x0
                            x1 = img.width - x1

                image_width = img.width
                image_height = img.height

                tmp = x0
                x0 = min(x0, x1)
                x1 = max(tmp, x1)

                tmp = y0
                y0 = min(y0, y1)
                y1 = max(tmp, y1)

                x0 = max(x0, 0)
                y0 = max(y0, 0)

                y0 = min(y0, image_height)
                x0 = min(x0, image_width)
                y1 = min(y1, image_height)
                x1 = min(x1, image_width)

                if self.rnd_color:
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(np.random.uniform(low=0.5, high=1.5))

                    enhancer2 = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(np.random.uniform(low=0.7, high=1.3))

                img = img.resize((self.image_size, self.image_size))
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32)

                if self.rnd_multiply:
                    img[...,0] = np.floor(np.clip(img[...,0] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))
                    img[...,1] = np.floor(np.clip(img[...,1] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))
                    img[...,2] = np.floor(np.clip(img[...,2] * np.random.uniform(low=0.8, high=1.2), 0.0, 255.0))

                batch_images[i] = preprocess_input(img.copy())

            x_c = ((self.grid_size - 1) / image_width) * (x0 + (x1 - x0) / 2)
            y_c = ((self.grid_size - 1) / image_height) * (y0 + (y1 - y0) / 2)

            floor_y = math.floor(y_c)
            floor_x = math.floor(x_c)

            batch_boxes[i, floor_y, floor_x, 0] = (y1 - y0) / image_height
            batch_boxes[i, floor_y, floor_x, 1] = (x1 - x0) / image_width
            batch_boxes[i, floor_y, floor_x, 2] = y_c - floor_y
            batch_boxes[i, floor_y, floor_x, 3] = x_c - floor_x
            batch_boxes[i, floor_y, floor_x, 4] = 1

            if self.debug:
                changed = img.astype(np.uint8)
                if not os.path.exists("__debug__"):
                    os.makedirs("__debug__")

                changed = Image.fromarray(changed)

                x_c = (floor_x + batch_boxes[i, floor_y, floor_x, 3]) / (self.grid_size - 1)
                y_c = (floor_y + batch_boxes[i, floor_y, floor_x, 2]) / (self.grid_size - 1)

                y0 = self.image_size * (y_c - batch_boxes[i, floor_y, floor_x, 0] / 2)
                x0 = self.image_size * (x_c - batch_boxes[i, floor_y, floor_x, 1] / 2)
                y1 = y0 + self.image_size * batch_boxes[i, floor_y, floor_x, 0]
                x1 = x0 + self.image_size * batch_boxes[i, floor_y, floor_x, 1]

                draw = ImageDraw.Draw(changed)
                draw.rectangle(((x0, y0), (x1, y1)), outline="green")

                changed.save(os.path.join("__debug__", os.path.basename(path)))
        #
        # for idx, image in enumerate(batch_images[:5]):
        #     pure_image = image
        #     box_image = cv2.rectangle(pure_image, (50, 50), (50 + 10, 50 + 20), (255, 255, 00), 2)
        #     pyplot.imshow(pure_image)
        #     pyplot.show()
        #     pyplot.imshow(box_image)
        #     pyplot.show()
        return batch_images, batch_boxes


class Validation(Callback):
    def get_box_highest_percentage(self, mask):
        reshaped = mask.reshape(mask.shape[0], np.prod(mask.shape[1:-1]), -1)

        score_ind = np.argmax(reshaped[...,-1], axis=-1)
        unraveled = np.array(np.unravel_index(score_ind, mask.shape[:-1])).T[:,1:]

        cell_y, cell_x = unraveled[...,0], unraveled[...,1]
        boxes = mask[np.arange(mask.shape[0]), cell_y, cell_x]

        h, w, offset_y, offset_x = boxes[...,0], boxes[...,1], boxes[...,2], boxes[...,3]

        return np.stack([cell_y + offset_y, cell_x + offset_x,
                        (self.grid_size - 1) * h, (self.grid_size - 1) * w], axis=-1)

    def __init__(self, generator, grid_size):
        self.generator = generator
        self.grid_size = grid_size

    def on_epoch_end(self, epoch, logs):
        mse = 0
        intersections = 0
        unions = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)

            pred = self.get_box_highest_percentage(pred)
            gt = self.get_box_highest_percentage(gt)

            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]

            pred = np.maximum(pred, 0)
            gt = np.maximum(gt, 0)

            diff_height = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
            diff_width = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:,2] * gt[:,3]
            area_pred = pred[:,2] * pred[:,3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse

        print(" - val_iou: {} - val_mse: {}".format(iou, mse))


class MobileNetV2Detector():
    # Default initialization
    # ?
    ALPHA = 0.35

    GRID_SIZE = 14
    # Input parameters configuration
    IMAGE_SIZE = 448
    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 640

    # Model training configuration
    EPOCHS = 200
    BATCH_SIZE = 32
    PATIENCE = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0005
    LR_DECAY = 0.0001

    # Retraining configuration
    TRAINABLE = False
    WEIGHTS = "model-0.27.h5"

    # Mltithreading computing params
    MULTITHREADING = False
    THREADS = 1

    # Path to input files
    TRAIN_CSV = "train.csv"
    VALIDATION_CSV = "validation.csv"

    # Detection model
    MODEL = None

    def __init__(self):
        with open('../configuration/detection.json', 'r') as f:
            # TODO model configuration initialization from detection.json
            model_conf = json.load(f)

        self.ALPHA = model_conf['ALPHA']

        self.GRID_SIZE = 14
        # Input parameters configuration
        self.IMAGE_SIZE = 448
        self.IMAGE_HEIGHT = 360
        self.IMAGE_WIDTH = 640

        # Model training configuration
        self.EPOCHS = 200
        self.BATCH_SIZE = 32
        self.PATIENCE = 15
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 0.0005
        self.LR_DECAY = 0.0001

        # Retraining configuration
        self.TRAINABLE = False
        self.WEIGHTS = model_conf['WEIGHTS']

        # self.Mltithreading computing params
        self.MULTITHREADING = False
        self.THREADS = 1

        # Path to input files
        self.TRAIN_CSV = "train.csv"
        self.VALIDATION_CSV = "validation.csv"

        self.init_model()

    def init_model(self, trainable=False):
        """
        :param trainable: using pretrained model configuration or not
        :param model_conf: loaded from json configuration model scheme
        :return:
        """

        model = MobileNetV2(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
                            include_top=False, alpha=self.ALPHA, weights='imagenet')

        for layer in model.layers:
            layer.trainable = trainable

        block = model.get_layer("block_16_project_BN").output

        x = Conv2D(112, padding="same", kernel_size=3, strides=1, activation="relu")(block)
        x = Conv2D(112, padding="same", kernel_size=3, strides=1, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(5, padding="same", kernel_size=1, activation="sigmoid")(x)

        model = Model(inputs=model.input, outputs=x)

        # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
        # see https://arxiv.org/pdf/1711.05101.pdf
        regularizer = l2(self.WEIGHT_DECAY / 2)
        for weight in model.trainable_weights:
            with tf.keras.backend.name_scope("weight_regularizer"):
                model.add_loss(regularizer(weight))

        self.MODEL = model

    def save_model_scheme(self):
        plot_model(MobileNetV2(input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
                               include_top=False, alpha=self.ALPHA, weights=None),
                   to_file='model.png', show_shapes=True)

    def train(self):
        with open('../configuration/detection.json', 'r') as f:
            model_conf = json.load(f)

        if self.TRAINABLE:
            self.MODEL.load_weights(self.WEIGHTS)

        train_datagen = DataGenerator(self.TRAIN_CSV, rnd_rescale=False, rnd_multiply=False, rnd_crop=False,
                                      rnd_flip=False, debug=False, batch_size=self.BATCH_SIZE,
                                      image_size=self.IMAGE_SIZE, grid_size=self.GRID_SIZE)

        val_generator = DataGenerator(self.VALIDATION_CSV, rnd_rescale=False, rnd_multiply=False, rnd_crop=False,
                                      rnd_flip=False, debug=False, batch_size=self.BATCH_SIZE,
                                      image_size=self.IMAGE_SIZE, grid_size=self.GRID_SIZE)

        validation_datagen = Validation(generator=val_generator, grid_size=self.GRID_SIZE)

        learning_rate = self.LEARNING_RATE
        if self.TRAINABLE:
            learning_rate /= 10

        optimizer = SGD(lr=learning_rate, decay=self.LR_DECAY, momentum=0.9, nesterov=False)
        self.MODEL.compile(loss=detection_loss(self.GRID_SIZE), optimizer=optimizer, metrics=[])

        checkpoint = ModelCheckpoint("results/model-{val_iou:.2f}.h5", monitor="val_iou", verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True, mode="max", period=1)
        stop = EarlyStopping(monitor="val_iou", patience=self.PATIENCE, mode="max")
        reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.6, patience=5, min_lr=1e-6, verbose=1, mode="max")

        self.MODEL.fit_generator(generator=train_datagen, epochs=self.EPOCHS,
                                 callbacks=[validation_datagen, checkpoint, reduce_lr, stop], shuffle=True, verbose=1
                                 # workers=THREADS, # use_multiprocessing=MULTITHREADING,
                                )
