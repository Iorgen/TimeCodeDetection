from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras import backend as K
from core.loss_func import ctc_lambda_func
from keras.optimizers import Adam


def init_recognition_model(model_conf):
    img_w = model_conf["IMAGE_WIDTH"]
    img_h = model_conf["IMAGE_HEIGHT"]
    absolute_max_string_len = model_conf["ABSOLUTE_MAX_STRING"]
    conv_filters = model_conf["CONV_FILTERS"]
    kernel_size = model_conf["KERNEL_SIZE"]
    pool_size = model_conf["POOL_SIZE"]
    time_dense_size = model_conf["TIME_DENSE_SIZE"]
    rnn_size = model_conf["RNN_SIZE"]


    if K.image_data_format() == 'channels_first':
        input_shape = (1, model_conf["IMAGE_WIDTH"],model_conf["IMAGE_HEIGHT"])
    else:
        input_shape = (model_conf["IMAGE_WIDTH"],model_conf["IMAGE_HEIGHT"], 1)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    # transforms RNN output to character activations:
    inner = Dense(len(model_conf["ALPHABET"]) + 1, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    # Model(inputs=input_data, outputs=y_pred).summary()
    labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    # LOSS FUNCTION NEED TO UNDERSTAND
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    model.summary()

    return model, input_data, y_pred
