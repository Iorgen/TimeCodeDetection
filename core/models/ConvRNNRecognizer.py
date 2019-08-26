import os
import itertools
import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras import backend as K
from core.loss_func import ctc_lambda_func
from keras.optimizers import Adam
from core.generators.recognition_generators import TimeCodeCropGenerator


class ConvRNNRecognitionModel():
    OUTPUT_DIR = 'logs'
    # Two color mode
    ALPHABET = ''
    input_data = None
    y_pred = None
    # Recognition model
    MODEL = None
    # Recognition model for inference
    inference_model = None

    def __init__(self, model_conf):
        self.ALPHABET = model_conf['ALPHABET']
        self.WORDS_PER_EPOCH = model_conf['WORDS_PER_EPOCH']
        self.VAL_SPLIT = model_conf['VAL_SPLIT']
        self.MINIBATCH_SIZE = model_conf['MINIBATCH_SIZE']
        self.IMAGE_WIDTH = model_conf['IMAGE_WIDTH']
        self.IMAGE_HEIGHT = model_conf['IMAGE_HEIGHT']
        self.POOL_SIZE = model_conf['POOL_SIZE']
        self.ABSOLUTE_MAX_STRING = model_conf['ABSOLUTE_MAX_STRING']
        self.CONV_FILTERS = model_conf['CONV_FILTERS']
        self.KERNEL_SIZE = model_conf['KERNEL_SIZE']
        self.TIME_DENSE_SIZE = model_conf['TIME_DENSE_SIZE']
        self.RNN_SIZE = model_conf['RNN_SIZE']
        self.WEIGHTS_DIR = model_conf['WEIGHTS_DIR']
        self.init_model()

    def make_predict(self, input):
        net_out_value = self.inference_model.predict(input)
        pred_texts = self.decode_predict_ctc(net_out_value)
        return pred_texts

    def init_model(self):

        if K.image_data_format() == 'channels_first':
            input_shape = (1,self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        else:
            input_shape = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1)

        act = 'relu'
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE,
                       padding='same', activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE, self.POOL_SIZE), name='max1')(inner)
        inner = Conv2D(self.CONV_FILTERS, self.KERNEL_SIZE, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.POOL_SIZE, self.POOL_SIZE), name='max2')(inner)
        conv_to_rnn_dims = (self.IMAGE_WIDTH // (self.POOL_SIZE ** 2), (self.IMAGE_HEIGHT //
                                                                        (self.POOL_SIZE ** 2)) * self.CONV_FILTERS)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
        # cuts down input size going into RNN:
        inner = Dense(self.TIME_DENSE_SIZE, activation=act, name='dense1')(inner)
        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        gru_1 = GRU(self.RNN_SIZE, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
        gru_1b = GRU(self.RNN_SIZE, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru1_b')(inner)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(self.RNN_SIZE, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(self.RNN_SIZE, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                     name='gru2_b')(gru1_merged)
        # transforms RNN output to character activations:
        inner = Dense(len(self.ALPHABET) + 1, kernel_initializer='he_normal',
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)
        # Model(inputs=input_data, outputs=y_pred).summary()
        labels = Input(name='the_labels', shape=[self.ABSOLUTE_MAX_STRING], dtype='float32')
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

        # get keras model structure
        self.MODEL = model
        self.input_data = input_data
        self.y_pred = y_pred
        self.inference_model = Model(inputs=self.input_data, outputs=self.y_pred)

    # Reverse translation of numerical classes back to characters
    def labels_to_text(self, labels):
        ret = []
        for c in labels:
            if c == len(self.ALPHABET):  # CTC Blank
                ret.append("")
            else:
                ret.append(self.ALPHABET[c])
        return "".join(ret)

    # For a real OCR application, this should be beam search with a dictionary
    # and language model.  For this example, best path is sufficient.
    def decode_batch(self, test_func, word_batch):
        out = test_func([word_batch])[0]
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = self.labels_to_text(out_best)
            ret.append(outstr)
        return ret

    def decode_predict_ctc(self, out, top_paths=1):
        results = []
        beam_width = 5
        if beam_width < top_paths:
            beam_width = top_paths
        for i in range(top_paths):
            lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
                                              greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
            text = self.labels_to_text(lables)
            results.append(text)
        return results

    def train(self, run_name, start_epoch, stop_epoch):
        val_words = int(self.WORDS_PER_EPOCH * (self.VAL_SPLIT))
        words_dir = os.path.join('train', 'recognition_dataset')
        # fdir = 'train/recognition_dataset'

        img_gen = TimeCodeCropGenerator(monogram_file=os.path.join(words_dir, 'wordlist_mono_clean.txt'),
                                        bigram_file=os.path.join(words_dir, 'wordlist_bi_clean.txt'),
                                        minibatch_size=self.MINIBATCH_SIZE,
                                        img_w=self.IMAGE_WIDTH,
                                        img_h=self.IMAGE_HEIGHT,
                                        downsample_factor=(self.POOL_SIZE ** 2),
                                        val_split=self.WORDS_PER_EPOCH - val_words, alphabet=self.ALPHABET)

        # preeducation condition
        if start_epoch > 0:
            weight_file = os.path.join(self.OUTPUT_DIR,
                                       os.path.join(run_name, 'recognition_weights%02d.h5' % (start_epoch - 1)))
            self.MODEL.load_weights(weight_file)

        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([self.input_data], [self.y_pred])

        # start model education
        self.MODEL.fit_generator(generator=img_gen.next_train(),
                                 steps_per_epoch=(self.WORDS_PER_EPOCH - val_words) // self.MINIBATCH_SIZE,
                                 epochs=stop_epoch,
                                 validation_data=img_gen.next_val(),  # on validation step
                                 validation_steps=val_words // self.MINIBATCH_SIZE,
                                 # callbacks=[viz_cb, img_gen],
                                 callbacks=[img_gen], initial_epoch=start_epoch)

        # save model weights
        self.MODEL.save_weights(os.path.join(self.WEIGHTS_DIR, 'model_weights.h5'))



