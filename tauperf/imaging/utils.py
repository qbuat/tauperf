import os
import tables
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from load import get_X_y
from . import log; log = log.getChild(__name__)

class DataSequence(Sequence):
    """
    See https://keras.io/utils/
    """
    def __init__(
            self, 
            train_indices,
            data_types,
            n_chunks, 
            features, 
            data_dir,
            predict=False,
            reg_features=None,
            equal_size=False, 
            debug=False):
        self._features = features
        self._reg_features = reg_features
        self._data_dir = data_dir
        self._n_chunks = n_chunks
        self._equal_size = equal_size
        self._debug = debug
        self._train_indices = train_indices
        self._data_types = data_types
        self._predict = predict

    def __len__(self):
        return self._n_chunks

    def __getitem__(self, idx):
        index = self._train_indices[idx]
        h5file = tables.open_file(os.path.join(
                self._data_dir, 'tables_{}.h5'.format(index)))
        X, y = get_X_y(
            h5file, 
            self._data_types, 
            equal_size=self._equal_size, 
            debug=self._debug)
        X_inputs = [X[feat] for feat in self._features]
        h5file.close()
        # if self._predict:
        #     return X_inputs
        # if self._reg_features is None:
        return X_inputs, to_categorical(y, len(self._data_types))
        # else:
        #     # outputs = [to_categorical(y, len(self._files))]
        #     # if not isinstance(self._reg_features, (list, tuple)):
        #     #     self._reg_features = [self._reg_features]
            
        #     # for feat in self._reg_features:
        #     #     outputs.append(X[feat])
        #     return X_inputs, X[self._reg_features[0]]

def fit_model_gen(
        model,
        data_dir,
        data_types,
        features,
        train_indices,
        valid_indices,
        reg_features=None,
        n_chunks=3,
        use_multiprocessing=False,
        workers=1,
        filename='cache/test.h5',
        metrics='categorical_accuracy',
        losses='categorical_crossentropy',
        model_optimizer='rmsprop',
        loss_weights=1,
        epochs=10,
        overwrite=False,
        no_train=False,
        equal_size=False,
        dev=False,
        debug=False):
    
 
    log.info('will write to {}'.format(filename))
    if not overwrite and os.path.exists(filename):
        log.error('weight file {0} exists, aborting!'.format(filename))
        raise ValueError('overwrite needs to be set to true')

    try:
        log.info('Compile model')
        if not isinstance(losses, (tuple, list)):
            losses = [losses]
        
        if not isinstance(metrics, (tuple, list)):
            metrics = [metrics]

        if not isinstance(loss_weights, (tuple, list)):
            loss_weights = [loss_weights]
        
        if reg_features != None:
            if not isinstance(reg_features, (tuple, list)):
                reg_features = [reg_features]
            

        model.compile(
            optimizer=model_optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics)

        log.info('Create the train sequence size = {}'.format(len(train_indices)))
        train_sequence = DataSequence(
            train_indices,
            data_types,
            n_chunks, 
            features, 
            data_dir,
            reg_features=reg_features, 
            debug=debug)
            
        log.info('Create the validation sequence')
        valid_sequence = DataSequence(
            valid_indices,
            data_types,
            len(valid_indices), 
            features, 
            data_dir,
            reg_features=reg_features, 
            debug=debug)

        # TODO
        # need to understand how to pass the validation sequence
        val, y_val = valid_sequence[0]

        # val = []
        # y_val = []
        # for (X, y) in valid_sequence:
        #     val.append(X)
        #     y_val.append(y)

        # val = np.concatenate(val)
        # y_val = np.concatenat(y_val)
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

        # validation_data = (
        #     [X_test[feat] for feat in features],
        #     y_test if reg_features is None else X_test[reg_features[0]])#[X_test[feat] for feat in reg_features])

        log.info('start generator')
        model.fit_generator(
            train_sequence,
            len(train_sequence),
            epochs=epochs,
            validation_data=(val, y_val),
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            callbacks=callbacks)

        # train_sequence.close()
        # valid_sequence.close()


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

