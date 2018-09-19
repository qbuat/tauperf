import os
from . import log; log = log.getChild(__name__)
import numpy as np
import tables

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from load import get_X_y

class TrainSequence(Sequence):
    """
    See https://keras.io/utils/
    """
    def __init__(
        self, filenames, features, n_chunks, 
        equal_size=False, debug=False):

        self._files = filenames
        self.features = features
        self.n_chunks = n_chunks
        self._equal_size = equal_size
        self._debug = debug

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        h5files = [tables.open_file(f) for f in self._files]
        X, y = get_X_y(
            h5files, 'train_{0}'.format(idx), 
            equal_size=self._equal_size, debug=self._debug)
        X = [X[feat] for feat in self.features]
        for f in h5files:
            f.close()
        return X, to_categorical(y, len(self._files))


def fit_model_gen(
    model,
    h5files, features,
    X_test, y_test, 
    n_chunks=3,
    use_multiprocessing=False,
    workers=1,
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False,
    equal_size=False,
    dev=False,
    debug=False):

 
    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to true')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
            metrics=[categorical_accuracy])

        log.info('Create the sequence')
        train_sequence = TrainSequence(h5files, features, n_chunks, debug=debug)
            
        log.info('Start training ...')

        # make a list of callbacks
        callbacks = []
        if dev:
            callbacks = [
                ModelCheckpoint(filename, monitor='val_loss', verbose=True)
                ]
        else:
            callbacks = [
                EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
                ModelCheckpoint(filename, monitor='val_loss', verbose=True, save_best_only=True)
                ]

        model.fit_generator(
            train_sequence,
            len(train_sequence),
            epochs=100,
            validation_data=(X_test, y_test),
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            callbacks=callbacks)

#         model.save(filename)
    

    except KeyboardInterrupt:
        print 'Ended early..'

def fit_model(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to true')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
            metrics=['accuracy'])

        log.info('Start training ...')
        model.fit(
            [
                # X_train['tracks'], 
                X_train['s1'], 
                X_train['s2'], 
                X_train['s3'], 
                X_train['s4'], 
                X_train['s5']
                ],
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(
                [
                    # X_test['tracks'], 
                    X_test['s1'], 
                    X_test['s2'], 
                    X_test['s3'], 
                    X_test['s4'], 
                    X_test['s5']
                    ], y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        print 'Ended early..'


def fit_model_single_layer(
    model,
    X_train, y_train, 
    X_test, y_test, 
    filename='cache/crackpot.h5',
    loss='binary_crossentropy',
    overwrite=False,
    no_train=False):

 
    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to false')

    try:
        log.info('Compile model')
        model.compile(
            optimizer='rmsprop',
            loss=loss,
            metrics=['accuracy'])

        log.info('Start training ...')
        log.info(X_train.shape)
        model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(
                X_test, y_test),
            callbacks=[
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(
                    filename, monitor='val_loss', 
                    verbose=True, save_best_only=True)
                ])

        model.save(filename)
    

    except KeyboardInterrupt:
        log.info('Ended early..')

