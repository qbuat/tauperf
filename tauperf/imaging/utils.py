import os
import tables
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

from load import get_X_y
from . import log; log = log.getChild(__name__)

class TrainSequence(Sequence):
    """
    See https://keras.io/utils/
    """
    def __init__(
            self, 
            filenames, 
            n_chunks, 
            features, 
            reg_features=None,
            equal_size=False, 
            debug=False):

        self._files = filenames
        self._features = features
        self._reg_features = reg_features
        self.n_chunks = n_chunks
        self._equal_size = equal_size
        self._debug = debug

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        h5files = [tables.open_file(f) for f in self._files]
        X, y = get_X_y(
            h5files, 
            'train_{0}'.format(idx), 
            equal_size=self._equal_size, 
            debug=self._debug)
        X_inputs = [X[feat] for feat in self._features]
        for f in h5files:
            f.close()
        if self._reg_features is None:
            return X_inputs, to_categorical(y, len(self._files))
        else:
            outputs = [to_categorical(y, len(self._files))]
            for feat in self._reg_features:
                outputs.append(X[feat])
            return X_inputs, outputs

def fit_model_gen(
        model,
        h5files, 
        features,
        X_test, y_test, 
        reg_features=None,
        n_chunks=3,
        use_multiprocessing=False,
        workers=1,
        filename='cache/crackpot.h5',
        metrics='categorical_accuracy',
        losses='categorical_crossentropy',
        loss_weights=1,
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
        if not isinstance(losses, (tuple, list)):
            losses = [losses]
        
        if not isinstance(metrics, (tuple, list)):
            metrics = [metrics]

        if not isinstance(loss_weights, (tuple, list)):
            loss_weights = [loss_weights]

        model.compile(
            optimizer='rmsprop',
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics)

        log.info('Create the sequence')
        train_sequence = TrainSequence(
            h5files, 
            n_chunks, 
            features, 
            reg_features=reg_features, 
            debug=debug)
            
        log.info('Start training ...')

        # make a list of callbacks
        callbacks = []
        if dev:
            callbacks = [
                ModelCheckpoint(filename, monitor='val_loss', verbose=True)
                ]
        else:
            callbacks = [
                # EarlyStopping(verbose=True, patience=10, monitor='val_decay_mode_loss'),
                ModelCheckpoint(filename, monitor='val_decay_mode_loss', verbose=True, save_best_only=True)
                ]

        validation_data = (
            [X_test[feat] for feat in features],
            y_test if reg_features is None else [y_test] + [X_test[feat] for feat in reg_features])


        model.fit_generator(
            train_sequence,
            len(train_sequence),
            epochs=100,
            validation_data=validation_data,
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

