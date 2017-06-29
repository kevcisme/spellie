# encoding: utf-8
'''
Created on Nov 26, 2015

@author: tal

Based in part on:
Learn math - https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py

See this: https://medium.com/@majortal/deep-spelling-9ffef96a24f6#.2c9pu8nlm
"""

Modified by Kevin Coyle

'''

import argparse
import numpy as np
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.layers import recurrent
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, LambdaCallback
from numpy.random import seed as random_seed
from numpy.random import randint as random_randint
import os
import pickle

from data import DataSet

random_seed(42)  # Reproducibility

# Parameters for the model and dataset
DATASET_FILENAME = 'data/dataset/news.2011.en.shuffled'
NUMBER_OF_EPOCHS = 2
RNN = recurrent.LSTM
INPUT_LAYERS = 2
OUTPUT_LAYERS = 2
AMOUNT_OF_DROPOUT = 0.3
BATCH_SIZE = 32
SAMPLES_PER_EPOCH = 65536
HIDDEN_SIZE = 700
INITIALIZATION = "he_normal"  # : Gaussian initialization scaled by fan_in (He et al., 2014)
NUMBER_OF_CHARS = 100  # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
INVERTED = True
MODEL_CHECKPOINT_DIRECTORYNAME = 'models'
MODEL_CHECKPOINT_FILENAME = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
MODEL_DATASET_PARAMS_FILENAME = 'dataset_params.pickle'
MODEL_STARTING_CHECKPOINT_FILENAME = 'weights.hdf5'
CSV_LOG_FILENAME = 'log.csv'


def generate_model(output_len, chars=None):
    """Generate the model"""
    print('Build model...')
    chars = chars or CHARS
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    for layer_number in range(INPUT_LAYERS):
        model.add(recurrent.LSTM(HIDDEN_SIZE, input_shape=(None, len(chars)), init=INITIALIZATION,
                                 return_sequences=layer_number + 1 < INPUT_LAYERS))
        model.add(Dropout(AMOUNT_OF_DROPOUT))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(output_len))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(OUTPUT_LAYERS):
        model.add(recurrent.LSTM(HIDDEN_SIZE, return_sequences=True, init=INITIALIZATION))
        model.add(Dropout(AMOUNT_OF_DROPOUT))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars), init=INITIALIZATION)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class Colors(object):
    """For nicer printouts"""
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def show_samples(model, dataset, epoch, logs, X_dev_batch, y_dev_batch):
    """Selects 10 samples from the dev set at random so we can visualize errors"""

    for _ in range(10):
        ind = random_randint(0, len(X_dev_batch))
        row_X, row_y = X_dev_batch[np.array([ind])], y_dev_batch[np.array([ind])]
        preds = model.predict_classes(row_X, verbose=0)
        q = dataset.character_table.decode(row_X[0])
        correct = dataset.character_table.decode(row_y[0])
        guess = dataset.character_table.decode(preds[0], calc_argmax=False)

        if INVERTED:
            print('Q', q[::-1])  # inverted back!
        else:
            print('Q', q)

        print('A', correct)
        print(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)
        print('---')



def iterate_training(model, dataset, initial_epoch):
    """Iterative Training"""

    checkpoint = ModelCheckpoint(MODEL_CHECKPOINT_DIRECTORYNAME + '/' + MODEL_CHECKPOINT_FILENAME,
                                 save_best_only=True)
    tensorboard = TensorBoard()
    csv_logger = CSVLogger(CSV_LOG_FILENAME)

    X_dev_batch, y_dev_batch = next(dataset.dev_set_batch_generator(1000))
    show_samples_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: show_samples(model, dataset, epoch, logs, X_dev_batch, y_dev_batch))

    train_batch_generator = dataset.train_set_batch_generator(BATCH_SIZE)
    validation_batch_generator = dataset.dev_set_batch_generator(BATCH_SIZE)

    model.fit_generator(train_batch_generator,
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=NUMBER_OF_EPOCHS,
                        validation_data=validation_batch_generator,
                        nb_val_samples=SAMPLES_PER_EPOCH,
                        callbacks=[checkpoint, tensorboard, csv_logger, show_samples_callback],
                        verbose=1,
                        initial_epoch=initial_epoch)


def save_dataset_params(dataset):
    params = { 'chars': dataset.chars, 'y_max_length': dataset.y_max_length }
    with open(MODEL_CHECKPOINT_DIRECTORYNAME + '/' + MODEL_DATASET_PARAMS_FILENAME, 'wb') as f:
        pickle.dump(params, f)


def main_news(checkpoint_filename=None, dataset_params_filename=None, initial_epoch=1):
    """Main"""
    dataset = DataSet(DATASET_FILENAME)

    if not os.path.exists(MODEL_CHECKPOINT_DIRECTORYNAME):
        os.makedirs(MODEL_CHECKPOINT_DIRECTORYNAME)

    if dataset_params_filename is not None:
        with open(dataset_params_filename, 'rb') as f:
            dataset_params = pickle.load(f)

        assert dataset_params['chars'] == dataset.chars
        assert dataset_params['y_max_length'] == dataset.y_max_length

    else:
        save_dataset_params(dataset)

    model = generate_model(dataset.y_max_length, dataset.chars)

    if checkpoint_filename is not None:
        model.load_weights(checkpoint_filename)

    iterate_training(model, dataset, initial_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a deep spelling model.')
    parser.add_argument('--checkpoint', type=str,
                        help='Filename of a model checkpoint to start the training from.')
    parser.add_argument('--datasetparams', type=str,
                        help='Filename of a file with dataset params to load for continuing model training.')
    parser.add_argument('--initialepoch', type=int,
                        help='Initial epoch parameter for continuing model training.', default=0)

    args = parser.parse_args()

    main_news(args.checkpoint, args.datasetparams, args.initialepoch)
