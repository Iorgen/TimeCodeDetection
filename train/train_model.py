from augmentation import *


# from core.recognition import augmentation,
def train(run_name, start_epoch, stop_epoch):
    # Input Parameters
    img_w = 135
    img_h = 35
    #     words_per_epoch = 2000
    words_per_epoch = 1600
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    minibatch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    fdir = '../dataset'
    img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                 bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words
                                 )
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
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    # Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    # LOSS FUNCTION NEED TO UNDERSTAND
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.summary()


    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)

    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])
    # viz_cb = VizCallback(run_name, test_func, img_gen.next_val())
    model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(), # on validation step
                        validation_steps=val_words // minibatch_size,
                        # callbacks=[viz_cb, img_gen],
                        callbacks=[img_gen],
                        initial_epoch=start_epoch)


if __name__ == '__main__':
    # fill the dataset with sequences
    # augmentation.create_sequence_dataset()
    h = 35
    w = 135
    generate_image = paint_text_cv('22:22:22', w, h)
    print(generate_image.shape)
    generate_image = generate_image.reshape(generate_image.shape[1], generate_image.shape[2])
    # cv2.imshow('images', generate_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    train(run_name, 0, 20)
    # TODO Save model inference for future versions of this model



