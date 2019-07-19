import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
# TODO create configuration file as on recognition model
IMAGE_SIZE = 224
WEIGHT_DECAY = 0.0005
ALPHA = 0.35


def init_detection_model(trainable=False):
    model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, alpha=ALPHA, weights="imagenet")

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
    regularizer = l2(WEIGHT_DECAY / 2)
    for weight in model.trainable_weights:
        with tf.keras.backend.name_scope("weight_regularizer"):
            model.add_loss(regularizer(weight))

    return model

