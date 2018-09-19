import os
import numpy as np
from . import log; log = log.getChild(__name__)
import tables


def print_sample_size(filenames, labels):
    if len(filenames) != len(labels):
        raise ValueError('filenames and labels must have the same length')

    trains = []
    tests  = []
    vals   = []
    for f in filenames:
        table = tables.open_file(f)
        test = len(table.root.data.test)
        val  = len(table.root.data.val)
        train = 0
        for obj in table.root.data:
            if isinstance(obj, tables.Table):
                if 'train_' in obj.name:
                    train += len(obj)

        table.close()
        trains.append(train)
        tests.append(test)
        vals.append(val)

    headers = ["Sample", "Training", "Validation", "Testing"]
    sample_size_table = []
    for l, tr, v, te in zip(labels, trains, vals, tests):
        sample_size_table.append([l, tr, v, te])

    log.info('Samples:')
    print 
    from tabulate import tabulate
    print tabulate(sample_size_table, headers=headers, tablefmt='simple')
    print


def get_X_y(h5_files, data_type, equal_size=False, debug=False):

    data = []
    for h5_file in h5_files:
        t = getattr(h5_file.root.data, data_type)
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


def load_test_data(filenames):

    h5files = [tables.open_file(filename) for filename in filenames]
    X_test, y_test = get_X_y(h5files, 'test')
    X_val, y_val = get_X_y(h5files, 'val')
    for f in h5files:
        f.close()
    return X_test, X_val, y_test, y_val
