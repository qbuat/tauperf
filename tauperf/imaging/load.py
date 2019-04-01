import os
import numpy as np
from . import log; log = log.getChild(__name__)
import tables
from sklearn import model_selection


def prepare_samples(filenames, labels):
    """
    printing information about the samples
    (training, validation and testing size)
    """
    log.info('Samples:')

    # if len(filenames) != len(labels):
    #     raise ValueError('filenames and labels must have the same length')

    # tables_size = []
    # n_tables = []
    # for f in filenames:
    #     file = tables.open_file(f)
    #     n_table_file = len(file.root.data)
        # sizes = []
        # n_table_file = 0
        # for obj in file.root.data:
        #     if isinstance(obj, tables.Table):
        #         n_table_file += 1
        #         sizes.append(len(obj))
        # tables_size.append(sizes)
        # n_tables.append(n_table_file)
        # file.close()

    # if not all(v == n_tables[0] for v in n_tables):
    #     raise ValueError('samples have different number of tables')

    # log.info('Number of tables for each sample: {}'.format(n_tables))
    # take 20% of the sample for validation and testing

    n_chunks = 1000
    train_ind, test_ind = model_selection.train_test_split(
        xrange(n_chunks), test_size=0.10, random_state=42)
    val_ind, test_ind = np.split(test_ind, [len(test_ind) / 2])

    n_training_chunks = n_chunks
    train_ind = train_ind[0:n_training_chunks]

    # headers = ["Sample", "Training", "Validation", "Testing", "Training tables"]
    # sample_size_table = []
    # for l, sizes in zip(labels, tables_size):
    #     train = sum([sizes[i] for i in train_ind])
    #     test  = sum([sizes[i] for i in test_ind])
    #     val   = sum([sizes[i] for i in val_ind])
    #     sample_size_table.append([l, train, val, test, len(train_ind)])

    # print 
    # from tabulate import tabulate
    # print tabulate(sample_size_table, headers=headers, tablefmt='simple')
    # print
    
    return train_ind, test_ind, val_ind

def get_X_y(h5_file, data_types, equal_size=False, debug=False):

    data = []
    for data_type in data_types:
        t = getattr(h5_file.root.data, 'table_{}'.format(data_type))
        data.append(t)
    
    if equal_size:
        log.info('Train and validate with equal size for each mode')
        min_size = min([len(t) for t in data])

    if debug:
        # log.info('Train with very small stat for debugging')
        min_size = min([len(t) for t in data] + [1000])
        
    if equal_size or debug:
        data = [t[0:min_size] for t in data]
    else:
        data = [t.read() for t in data]

    X_data = np.concatenate([d for d in data])
    y_data = np.concatenate([d['truthmode'] for d in data])

    return X_data, y_data


def load_data(data_dir, data_types, test_indices, debug=False):


    log.info('loading test data: {}'.format(test_indices))

    X_test = []
    y_test = []
    for i_ind, index in enumerate(test_indices):
        h5file = tables.open_file(os.path.join(data_dir, 'tables_{}.h5'.format(index)))
        print i_ind, len(test_indices)
        X, y = get_X_y(h5file, data_types, debug=debug)
        X_test.append(X)
        y_test.append(y)
        h5file.close()

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
                             
    return X_test, y_test
