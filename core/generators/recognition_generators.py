import os
import codecs
import cv2
import keras.callbacks
import numpy as np
from random import randint, choice, uniform
from random import randint, randrange
from glob import glob
from keras import backend as K


class TimeCodeCropGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split, alphabet,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.alphabet = alphabet
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(self.alphabet) + 1

    # Call only once a time at the start of education and after that
    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        # monogram file is sorted by frequency in english speech
        with codecs.open(self.monogram_file, mode='r', encoding='utf-8') as f:
            for line in f:
                if len(tmp_string_list) == int(self.num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)

        # bigram file contains common word pairings in english speech
        with codecs.open(self.bigram_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == self.num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if self.is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        #         print(tmp_string_list)
        if len(tmp_string_list) != self.num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
        # interlace to mix up the easy and hard words
        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = self.text_to_labels(word)
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an images is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    # on each batch in current epoch return data for train - non parallel process
    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = self.shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    # on validation step on each epoch - paralell process
    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(16000, 10, 1)
        self.paint_func = lambda text: self.paint_text_cv(text, self.img_w, self.img_h)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if 3 <= epoch < 6:
            self.paint_func = lambda text: self.paint_text_cv(text, self.img_w, self.img_h)
        elif 6 <= epoch < 9:
            self.paint_func = lambda text: self.paint_text_cv(text, self.img_w, self.img_h)
        elif epoch >= 9:
            self.paint_func = lambda text: self.paint_text_cv(text, self.img_w, self.img_h)
        if epoch >= 21 and self.max_string_len < 12:
            self.build_word_list(8000, 11, 0.5)

    # change or cut off this function from algorithm
    def is_valid_str(self, in_str):
        #     search = re.compile(regex, re.UNICODE).search
        #     return bool(search(in_str))
        return bool(True)

    # Translation of characters to unique integer values
    def text_to_labels(self, text):
        ret = []
        for char in text:
            ret.append(self.alphabet.find(char))
        return ret

    @staticmethod
    def paint_text_cv(input_text, width, height, sample_image_path=os.path.join('train', 'recognition_dataset', '*jpg')):
        ''' Function which put text on image
        :param input_text:  text to be placed on the images
        :param width: ended width of generated images
        :param height: ended height of generated images
        :param sample_image_path: path to images samples on which overlay occurs (default: "datasets/train_data/*jpg")
        :return: np.array()
        '''

        COLOR_SET = [(20, 20, 20), (230, 230, 230)]
        # TODO upload another fonts and generate text based on them
        FONT_SET = [0, 1, 2, 3, 4, 5, 6, 7, 16]

        # Set characteristics of text
        font = choice(FONT_SET)
        bottom_left_corner = (7 + randint(-4, 4), 20 + randint(-5, 5))  # That parameter should be random set
        font_scale = round(uniform(1.5, 2), 1)
        font_color = choice(COLOR_SET)
        line_thickness = 2

        # choose random image from dataset directory
        imgs = glob(sample_image_path)
        img_num = randrange(len(imgs))
        img = cv2.imread(imgs[img_num], 0)

        # print text on image
        cv2.putText(img,
                    input_text,
                    bottom_left_corner,
                    font,
                    font_scale,
                    font_color,
                    line_thickness)

        img = cv2.resize(img, (width, height))
        return img.reshape(1, height, width)

    @staticmethod
    def shuffle_mats_or_lists(matrix_list, stop_ind=None):
        ret = []
        assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
        len_val = len(matrix_list[0])
        if stop_ind is None:
            stop_ind = len_val
        assert stop_ind <= len_val

        a = list(range(stop_ind))
        np.random.shuffle(a)
        a += list(range(stop_ind, len_val))
        for mat in matrix_list:
            if isinstance(mat, np.ndarray):
                ret.append(mat[a])
            elif isinstance(mat, list):
                ret.append([mat[i] for i in a])
            else:
                raise TypeError('`shuffle_mats_or_lists` only supports '
                                'numpy.array and list objects.')
        return ret
