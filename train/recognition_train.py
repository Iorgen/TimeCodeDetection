# import os
# import itertools
# import codecs
# import datetime
# import editdistance
# import cv2
# import json
# import keras.callbacks
# import numpy as np
# import matplotlib.pyplot as plt
# from random import randint, choice, uniform
# from matplotlib import pylab
# from random import randint, randrange
# from glob import glob
# from keras import backend as K
# from core.recognition_model import init_recognition_model


# OUTPUT_DIR = 'logs'
# # Two color mode
# COLOR_SET = [(20, 20, 20), (230, 230, 230)]
# # TODO upload another fonts and generate text based on them
# FONT_SET = [0, 1, 2, 3, 4, 5, 6, 7, 16]
#
#
# with open('configuration/recognition.json', 'r') as f:
#     model_conf = json.load(f)
#     regex = model_conf['regex']
#     alphabet = model_conf['alphabet']


# def create_sequence_dataset():
#     with open('wordlist_mono_clean.txt', "w") as file:
#         sequence = []
#         for i in range(500000):
#             text = str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(
#                 randint(0, 9)) + str(randint(0, 9)) + '\n'
#             sequence.append(text)
#         file.writelines(sequence)
#
#     # fill the dataset with pair of sequences
#     with open('wordlist_bi_clean.txt', "w") as file:
#         sequence = []
#         for i in range(500000):
#             text = str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(
#                 randint(0, 9)) + str(randint(0, 9)) + ' ' + str(randint(0, 9)) + str(randint(0, 9)) + ':' + str(
#                 randint(0, 9)) + str(randint(0, 9)) + ':' + str(randint(0, 9)) + str(randint(0, 9)) + '\n'
#             sequence.append(text)
#         file.writelines(sequence)
#     import tarfile
#
#     def make_tarfile(output_filename, source_dir):
#         with tarfile.open(output_filename, "w:gz") as tar:
#             tar.add('wordlist_bi_clean.txt', arcname=os.path.basename("wordlist_bi_clean.txt"))
#             tar.add('wordlist_mono_clean.txt', arcname=os.path.basename("wordlist_mono_clean.txt"))
#
#     make_tarfile("wordlists.tgz", 'wordlists')


# def paint_text_cv(input_text, width, height, sample_image_path='recognition_dataset/*jpg'):
#     '''
#     :param input_text:  text to be placed on the images
#     :param width: ended width of generated images
#     :param height: ended height of generated images
#     :param sample_image_path: path to images samples on which the overlay occurs (default: "datasets/train_data/*jpg")
#     :return: np.array()
#     '''
#
#     # Set characteristics of text
#     font = choice(FONT_SET)
#     bottom_left_corner = (7 + randint(-4, 4), 20 + randint(-5, 5))  # That parameter should be random set
#     font_scale = round(uniform(2.5, 3), 1)
#     font_color = choice(COLOR_SET)
#     line_thickness = 2
#
#     # choose random image from dataset directory
#     imgs = glob(sample_image_path)
#     img_num = randrange(len(imgs))
#     img = cv2.imread(imgs[img_num], 0)
#
#     # print text on image
#     cv2.putText(img,
#                 input_text,
#                 bottom_left_corner,
#                 font,
#                 font_scale,
#                 font_color,
#                 line_thickness)
#
#     img = cv2.resize(img, (width, height))
#     return img.reshape(1, height, width)

#
# def shuffle_mats_or_lists(matrix_list, stop_ind=None):
#     ret = []
#     assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
#     len_val = len(matrix_list[0])
#     if stop_ind is None:
#         stop_ind = len_val
#     assert stop_ind <= len_val
#
#     a = list(range(stop_ind))
#     np.random.shuffle(a)
#     a += list(range(stop_ind, len_val))
#     for mat in matrix_list:
#         if isinstance(mat, np.ndarray):
#             ret.append(mat[a])
#         elif isinstance(mat, list):
#             ret.append([mat[i] for i in a])
#         else:
#             raise TypeError('`shuffle_mats_or_lists` only supports '
#                             'numpy.array and list objects.')
#     return ret
#
#
# # Translation of characters to unique integer values
# def text_to_labels(text):
#     ret = []
#     for char in text:
#         ret.append(alphabet.find(char))
#     return ret
#
#
# # Reverse translation of numerical classes back to characters
# def labels_to_text(labels):
#     ret = []
#     for c in labels:
#         if c == len(alphabet):  # CTC Blank
#             ret.append("")
#         else:
#             ret.append(alphabet[c])
#     return "".join(ret)
#
# # only a-z and space..probably not to difficult
# # to expand to uppercase and symbols yea not a problem
#
# # change or cut off this function from algorithm
# def is_valid_str(in_str):
# #     search = re.compile(regex, re.UNICODE).search
# #     return bool(search(in_str))
#     return bool(True)


# # Uses generator functions to supply train/test with
# # data. Image renderings are text are created on the fly
# # each time with random perturbations
# class TextImageGenerator(keras.callbacks.Callback):
#
#     def __init__(self, monogram_file, bigram_file, minibatch_size,
#                  img_w, img_h, downsample_factor, val_split,
#                  absolute_max_string_len=16):
#
#         self.minibatch_size = minibatch_size
#         self.img_w = img_w
#         self.img_h = img_h
#         self.monogram_file = monogram_file
#         self.bigram_file = bigram_file
#         self.downsample_factor = downsample_factor
#         self.val_split = val_split
#         self.blank_label = self.get_output_size() - 1
#         self.absolute_max_string_len = absolute_max_string_len
#
#     def get_output_size(self):
#         return len(alphabet) + 1
#
#     # Call only once a time at the start of education and after that
#     # num_words can be independent of the epoch size due to the use of generators
#     # as max_string_len grows, num_words can grow
#     def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
#         assert max_string_len <= self.absolute_max_string_len
#         assert num_words % self.minibatch_size == 0
#         assert (self.val_split * num_words) % self.minibatch_size == 0
#         self.num_words = num_words
#         self.string_list = [''] * self.num_words
#         tmp_string_list = []
#         self.max_string_len = max_string_len
#         self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
#         self.X_text = []
#         self.Y_len = [0] * self.num_words
#
#         # monogram file is sorted by frequency in english speech
#         with codecs.open(self.monogram_file, mode='r', encoding='utf-8') as f:
#             for line in f:
#                 if len(tmp_string_list) == int(self.num_words * mono_fraction):
#                     break
#                 word = line.rstrip()
#                 if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
#                     tmp_string_list.append(word)
#
#         # bigram file contains common word pairings in english speech
#         with codecs.open(self.bigram_file, mode='r', encoding='utf-8') as f:
#             lines = f.readlines()
#             for line in lines:
#                 if len(tmp_string_list) == self.num_words:
#                     break
#                 columns = line.lower().split()
#                 word = columns[0] + ' ' + columns[1]
#                 if is_valid_str(word) and \
#                         (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
#                     tmp_string_list.append(word)
# #         print(tmp_string_list)
#         if len(tmp_string_list) != self.num_words:
#             raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
#         # interlace to mix up the easy and hard words
#         self.string_list[::2] = tmp_string_list[:self.num_words // 2]
#         self.string_list[1::2] = tmp_string_list[self.num_words // 2:]
#
#         for i, word in enumerate(self.string_list):
#             self.Y_len[i] = len(word)
#             self.Y_data[i, 0:len(word)] = text_to_labels(word)
#             self.X_text.append(word)
#         self.Y_len = np.expand_dims(np.array(self.Y_len), 1)
#
#         self.cur_val_index = self.val_split
#         self.cur_train_index = 0
#
#     # each time an images is requested from train/val/test, a new random
#     # painting of the text is performed
#     def get_batch(self, index, size, train):
#         # width and height are backwards from typical Keras convention
#         # because width is the time dimension when it gets fed into the RNN
#         if K.image_data_format() == 'channels_first':
#             X_data = np.ones([size, 1, self.img_w, self.img_h])
#         else:
#             X_data = np.ones([size, self.img_w, self.img_h, 1])
#
#         labels = np.ones([size, self.absolute_max_string_len])
#         input_length = np.zeros([size, 1])
#         label_length = np.zeros([size, 1])
#         source_str = []
#         for i in range(size):
#             # Mix in some blank inputs.  This seems to be important for
#             # achieving translational invariance
#             if train and i > size - 4:
#                 if K.image_data_format() == 'channels_first':
#                     X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
#                 else:
#                     X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
#                 labels[i, 0] = self.blank_label
#                 input_length[i] = self.img_w // self.downsample_factor - 2
#                 label_length[i] = 1
#                 source_str.append('')
#             else:
#                 if K.image_data_format() == 'channels_first':
#                     X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
#                 else:
#                     X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
#                 labels[i, :] = self.Y_data[index + i]
#                 input_length[i] = self.img_w // self.downsample_factor - 2
#                 label_length[i] = self.Y_len[index + i]
#                 source_str.append(self.X_text[index + i])
#         inputs = {'the_input': X_data,
#                   'the_labels': labels,
#                   'input_length': input_length,
#                   'label_length': label_length,
#                   'source_str': source_str  # used for visualization only
#                   }
#         outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
#         return (inputs, outputs)
#
#     # on each batch in current epoch return data for train - non parallel process
#     def next_train(self):
#         while 1:
#             ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
#             self.cur_train_index += self.minibatch_size
#             if self.cur_train_index >= self.val_split:
#                 self.cur_train_index = self.cur_train_index % 32
#                 (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
#                     [self.X_text, self.Y_data, self.Y_len], self.val_split)
#             yield ret
#
#     # on validation step on each epoch - paralell process
#     def next_val(self):
#         while 1:
#             ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
#             self.cur_val_index += self.minibatch_size
#             if self.cur_val_index >= self.num_words:
#                 self.cur_val_index = self.val_split + self.cur_val_index % 32
#             yield ret
#
#     def on_train_begin(self, logs={}):
#         self.build_word_list(16000, 10, 1)
#         self.paint_func = lambda text: paint_text_cv(text, self.img_w, self.img_h)
#
#     def on_epoch_begin(self, epoch, logs={}):
#         # rebind the paint function to implement curriculum learning
#         if 3 <= epoch < 6:
#             self.paint_func = lambda text: paint_text_cv(text, self.img_w, self.img_h)
#         elif 6 <= epoch < 9:
#             self.paint_func = lambda text: paint_text_cv(text, self.img_w, self.img_h)
#         elif epoch >= 9:
#             self.paint_func = lambda text: paint_text_cv(text, self.img_w, self.img_h)
#         if epoch >= 21 and self.max_string_len < 12:
#             self.build_word_list(8000, 11, 0.5)
#

# # For a real OCR application, this should be beam search with a dictionary
# # and language model.  For this example, best path is sufficient.
# def decode_batch(test_func, word_batch):
#     out = test_func([word_batch])[0]
#     ret = []
#     for j in range(out.shape[0]):
#         out_best = list(np.argmax(out[j, 2:], 1))
#         out_best = [k for k, g in itertools.groupby(out_best)]
#         outstr = labels_to_text(out_best)
#         ret.append(outstr)
#     return ret


# # ----------------------------------------------------------------
# # class only for create a vizualization after each epoch education
# class VizCallback(keras.callbacks.Callback):
#
#     def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
#         self.test_func = test_func
#         self.output_dir = os.path.join(
#             OUTPUT_DIR, run_name)
#         self.text_img_gen = text_img_gen
#         self.num_display_words = num_display_words
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)
#
#     def show_edit_distance(self, num):
#         num_left = num
#         mean_norm_ed = 0.0
#         mean_ed = 0.0
#         while num_left > 0:
#             word_batch = next(self.text_img_gen)[0]
#             num_proc = min(word_batch['the_input'].shape[0], num_left)
#             decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
#             for j in range(num_proc):
#                 edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
#                 mean_ed += float(edit_dist)
#                 mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
#             num_left -= num_proc
#         mean_norm_ed = mean_norm_ed / num
#         mean_ed = mean_ed / num
#         print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
#               % (num, mean_ed, mean_norm_ed))
#
#     def on_epoch_end(self, epoch, logs={}):
#         self.model.save_weights(os.path.join(self.output_dir, 'recognition_weights%02d.h5' % (epoch)))
#         self.show_edit_distance(256)
#         word_batch = next(self.text_img_gen)[0]
#         res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
#         if word_batch['the_input'][0].shape[0] < 256:
#             cols = 2
#         else:
#             cols = 1
#         for i in range(self.num_display_words):
#             plt.subplot(self.num_display_words // cols, cols, i + 1)
#             if K.image_data_format() == 'channels_first':
#                 the_input = word_batch['the_input'][i, 0, :, :]
#             else:
#                 the_input = word_batch['the_input'][i, :, :, 0]
#             plt.imshow(the_input.T, cmap='Greys_r')
#             plt.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
#         fig = pylab.gcf()
#         fig.set_size_inches(10, 13)
#         plt.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
#         plt.close()


# def decode_predict_ctc(out, top_paths=1):
#     results = []
#     beam_width = 5
#     if beam_width < top_paths:
#         beam_width = top_paths
#     for i in range(top_paths):
#         lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
#                                           greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
#         text = labels_to_text(lables)
#         results.append(text)
#     return results


# def train(run_name, start_epoch, stop_epoch):
#
#     val_words = int(model_conf["words_per_epoch"] * (model_conf["val_split"]))
#     fdir = 'recognition_dataset'
#
#     img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
#                                  bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
#                                  minibatch_size=model_conf["minibatch_size"],
#                                  img_w=model_conf["img_w"],
#                                  img_h=model_conf["img_h"],
#                                  downsample_factor=(model_conf["pool_size"] ** 2),
#                                  # TODO have a look inside original model and check that parameter
#                                  val_split=model_conf["words_per_epoch"] - val_words
#                                  )
#     # get keras model structure
#     model, input_data, y_pred = init_recognition_model(model_conf)
#
#     # preeducation condition
#     if start_epoch > 0:
#         weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'recognition_weights%02d.h5' % (start_epoch - 1)))
#         model.load_weights(weight_file)
#
#     # captures output of softmax so we can decode the output during visualization
#     test_func = K.function([input_data], [y_pred])
#
#     # start model education
#     model.fit_generator(generator=img_gen.next_train(),
#                         steps_per_epoch=(model_conf['words_per_epoch'] - val_words) // model_conf["minibatch_size"],
#                         epochs=stop_epoch,
#                         validation_data=img_gen.next_val(), # on validation step
#                         validation_steps=val_words // model_conf["minibatch_size"],
#                         # callbacks=[viz_cb, img_gen],
#                         callbacks=[img_gen],
#                         initial_epoch=start_epoch)
#
#     # save model weights
#     model.save_weights(os.path.join(model_conf["model_weigths_dir"], 'model_weights.h5'))


# if __name__ == '__main__':
#     run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
#     train(run_name, 0, 1)
from core.models.ConvRNNRecognizer import ConvRNNRecognitionModel
import datetime


if __name__=="__main__":
    detector = ConvRNNRecognitionModel()
    detector.MODEL.summary()
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    # train(run_name, 0, 1)
    detector.train(run_name, 0, 1)
    # detector.save_model_scheme()



