import tables
import os

folder = '/data/qbuat/IMAGING/v14/test_aod_1'

files = [
    'images_new_1p0n.h5',
    'images_new_1p1n.h5',
    'images_new_1p2n.h5',
    'images_new_3p0n.h5',
    'images_new_3p1n.h5',
]

lengths = {}
for f_name in files:
    print 'computing length of arrays in ', f_name
    f = tables.open_file(os.path.join(folder, f_name))
    lengths[f_name] = [len(t) for t in f.root.data]
    f.close()

for a, b, c, d in zip([lengths[f] for f in files]):
    print a, b, c, d
