import os
import numpy as np
from argparse import ArgumentParser
from keras.utils.np_utils import to_categorical
from tauperf.imaging.load import load_data, prepare_samples
from tauperf.imaging.utils import get_X_y
import tables
import logging

log = logging.getLogger(os.path.basename(__file__))
log.setLevel(logging.INFO)

parser = ArgumentParser()
parser.add_argument(
    '--no-train', default=False, action='store_true')
parser.add_argument(
    '--overwrite', default=False, action='store_true')
parser.add_argument(
    '--equal-size', default=False, action='store_true')
parser.add_argument(
    '--debug', default=False, action='store_true')
parser.add_argument(
    '--training-chunks', default=3, type=int)
parser.add_argument(
    '--one-prong-only', default=False, action='store_true')
parser.add_argument(
    '--dev', default=False, action='store_true')

args = parser.parse_args()

if args.debug:
    log.warning('')
    log.warning('DEBUG MODE ACTIVATED')
    log.warning('')

### location of the training data
data_dir = os.path.join(os.getenv('DATA_AREA'), 'v14/test_aod_2')
                        
if args.one_prong_only:
    data_types = ['1p0n', '1p1n', '1p2n']
    labels = ['1p0n', '1p1n', '1pXn']
    n_classes = 3
else: 
    data_types = ['1p0n', '1p1n', '1p2n', '3p0n', '3p1n']
    labels = ['1p0n', '1p1n', '1pXn', '3p0n', '3pXn']
    n_classes = 5


train_ind, test_ind, valid_ind = prepare_samples(args.training_chunks)
test_ind = test_ind[0:200]

#kine_features = ['pt', 'eta', 'phi']
#features = kine_features + ['tracks', 's1', 's2', 's3', 's4', 's5']
#reg_features = ['true_pt', 'true_eta', 'true_phi', 'true_m']

features = ['tracks', 's1', 's2', 's3', 's4', 's5']

val_file = tables.open_file(os.path.join(data_dir, 'tables_0.h5'))
val, _ = get_X_y(val_file, ['1p0n'])
val_file.close()


# ##############################################
if args.dev:
    model_filename = 'cache/multi_{0}_classes_{1}_chunks'.format(
        n_classes, args.training_chunks)
    model_filename += '_{epoch:02d}_epochs.h5'
else:
    model_filename = 'cache/multi_{0}_classes_bla.h5'.format(n_classes)

if args.no_train:
    log.info('loading model')
    from keras.models import load_model
    model = load_model(model_filename)
else:
    log.info('training...')
    from tauperf.imaging.models import *

    model = dense_merged_model_topo(val, n_classes=n_classes, final_activation='softmax')
    # model = dense_merged_model_topo_with_regression(test, n_classes=n_classes, final_activation='softmax')
    # model = dense_merged_model_multi_channels(test, n_classes=n_classes, final_activation='softmax')
    # model = dense_merged_model_topo_upsampled(test, n_classes=n_classes, final_activation='softmax')

    from tauperf.imaging.utils import fit_model_gen

#    metrics = ['categorical_accuracy'] + 4 * ['mean_squared_error']
#    losses = ['categorical_crossentropy'] + 4 * ['mean_squared_error']
#    loss_weights = [1., 0.1, 0.1, 0.1, 0.1]

    metrics = 'categorical_accuracy' 
    losses = 'categorical_crossentropy'
    loss_weights = 1.
  
    fit_model_gen(
        model,
        data_dir,
        data_types,
        features,
        train_ind,
        valid_ind,
        reg_features=None,
        n_chunks=len(train_ind),
        use_multiprocessing=False,
        workers=1,
        filename=model_filename,
        metrics=metrics,
        losses=losses,
        loss_weights=loss_weights,
        overwrite=args.overwrite,
        no_train=args.no_train, 
        equal_size=args.equal_size, 
        dev=args.dev,
        debug=args.debug)




# ##############################################
log.info('testing stuff')

log.info('compute classifier scores')
# test, y_test = load_data(
#     filenames, test_ind, debug=args.debug)

from tauperf.imaging.utils import DataSequence
test_sequence = DataSequence(
    test_ind,
    data_types,
    len(test_ind),
    features,
    data_dir,
    predict=True)

log.info('define test sequence')
#X_test  = [test[feat] for feat in features]
y_pred = model.predict_generator(
    test_sequence, 
    # batch_size=32, 
    verbose=1)

log.info('load test data')
test, y_test = load_data(
    data_dir, data_types, test_ind, debug=args.debug)


log.info('drawing the computer-vision confusion matrix')
from sklearn.metrics import confusion_matrix
from tauperf.imaging.plotting import plot_confusion_matrix


cnf_mat = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))

from tauperf.imaging.plotting import plot_confusion_matrix, plot_roc
plot_confusion_matrix(
    cnf_mat, classes=labels, 
    title='Confusion matrix, diagonal = {0:1.2f} %'.format(100 * diagonal),
    name='plots/imaging/confusion_matrix_categorical.pdf')

log.info('drawing the pantau confusion matrix')
cnf_mat = confusion_matrix(y_test[test['pantau'] < 5], test['pantau'][test['pantau'] < 5])
diagonal = float(np.trace(cnf_mat)) / float(np.sum(cnf_mat))
plot_confusion_matrix(
    cnf_mat, classes=labels, 
    title='Pantau confusion matrix, diagonal = {0:1.2f} %'.format(100 * diagonal),
    name='plots/imaging/confusion_matrix_reference.pdf')

log.info('drawing the roc curves and pantau WP')
from sklearn.metrics import roc_curve
from tauperf.imaging.plotting import plot_roc
plot_roc(y_test, y_pred, test['pantau'])

log.info('drawing the scores')
from tauperf.imaging.plotting import plot_scores
plot_scores(y_pred, y_test)

log.info('accuracy plot')
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
sns.regplot(x=test['pt'], y=np.argmax(y_pred, axis=1)-y_test, x_bins=10, fit_reg=None)
plt.savefig('plots/imaging/accuracy_pt.pdf')

log.info('job finished succesfully!')

log.info("copying over to quentin's eos")
import subprocess
subprocess.call('cp -r plots/imaging/* /eos/user/q/qbuat/imaging_dev/', shell=True)
