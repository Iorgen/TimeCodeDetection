import json
import sys
import numpy as np
import tensorflow as tf
# - For windows path bugs
sys.path.append(sys.path[0] + "/..")
from keras.layers import Conv2D
from keras.utils import plot_model
from core.loss_func import detection_loss
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import epsilon
from core.generators.timecode_image import TimeCodeImageGenerator


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
        # os path refactor
        with open('../configuration/detection.json', 'r') as f:
            # TODO model configuration initialization from detection.json
            model_conf = json.load(f)

        self.ALPHA = model_conf['ALPHA']

        self.GRID_SIZE = model_conf['GRID_SIZE']
        # Input parameters configuration
        self.IMAGE_SIZE = model_conf['IMAGE_SIZE']
        self.IMAGE_HEIGHT = model_conf['IMAGE_HEIGHT']
        self.IMAGE_WIDTH = model_conf['IMAGE_WIDTH']

        # Model training configuration
        self.EPOCHS = model_conf['EPOCHS']
        self.BATCH_SIZE = model_conf['BATCH_SIZE']
        self.PATIENCE = model_conf['PATIENCE']
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = model_conf['WEIGHT_DECAY']
        self.LR_DECAY = model_conf['LR_DECAY']

        # Retraining configuration
        self.TRAINABLE = model_conf['TRAINABLE']
        self.WEIGHTS = model_conf['WEIGHTS']

        # self.Multithreading computing params
        self.MULTITHREADING = model_conf['MULTITHREADING']
        self.THREADS = model_conf['THREADS']

        # Path to input files
        self.TRAIN_CSV = "detection_dataset/train.csv"
        self.VALIDATION_CSV = "detection_dataset/validation.csv"

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

        if self.TRAINABLE:
            self.MODEL.load_weights(self.WEIGHTS)

        train_datagen = TimeCodeImageGenerator(self.TRAIN_CSV, rnd_rescale=False, rnd_multiply=False, rnd_crop=False,
                                      rnd_flip=False, debug=False, batch_size=self.BATCH_SIZE,
                                      image_size=self.IMAGE_SIZE, grid_size=self.GRID_SIZE)

        val_generator = TimeCodeImageGenerator(self.VALIDATION_CSV, rnd_rescale=False, rnd_multiply=False, rnd_crop=False,
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
