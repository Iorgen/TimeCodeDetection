import os
import sys
import tensorflow as tf
import copy
from keras.utils import plot_model
from core.loss_func import detection_loss
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import *
from keras.regularizers import l2
from keras.optimizers import SGD
from core.generators.detection_generatos import TimeCodeImageGenerator
from core.callbacks.detection_callbacks import Validation
# - For windows path bugs
sys.path.append(sys.path[0] + "/..")


class MobileNetV2Detector():
    # Default initialization
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

    def __init__(self, model_conf):
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

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

        # CSV files
        self.TRAIN_CSV = os.path.join('train', 'detection_dataset', model_conf['TRAIN_CSV'])
        self.VALIDATION_CSV = os.path.join('train', 'detection_dataset', model_conf['VALIDATION_CSV'])

        self.init_model()

    def get_model(self):
        return copy.deepcopy(self.MODEL)

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

        checkpoint = ModelCheckpoint(os.path.join("train", "results", "model-{val_iou:.2f}.h5"), monitor="val_iou", verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True, mode="max", period=1)
        stop = EarlyStopping(monitor="val_iou", patience=self.PATIENCE, mode="max")
        reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.6, patience=5, min_lr=1e-6, verbose=1, mode="max")

        self.MODEL.fit_generator(generator=train_datagen, epochs=self.EPOCHS,
                                 callbacks=[validation_datagen, checkpoint, reduce_lr, stop], shuffle=True, verbose=1
                                 # workers=THREADS, # use_multiprocessing=MULTITHREADING,
                                )
