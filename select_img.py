
import os
import numpy as np
from h5py import File
import tables
#from tauperf import log; log = log['/select-img']
from tauperf.imaging.processing import process_taus

import logging
log = logging.getLogger(os.path.basename(__file__))
log.setLevel(logging.INFO)

tau_type = '1p2n'
h5_filename = os.path.join(
    os.getenv('DATA_AREA'), 'v13', 'test', 'output.selected.h5')

# h5file = File(h5_filename, mode='r')
# rec = h5file.get('rec_' + tau_type)

h5file = tables.open_file(h5_filename)
rec = getattr(h5file.root,  'tree_' + tau_type)





# bad_img = 0

# for ix in xrange(len(rec_1p1n)):
#     rec = rec_1p1n[ix]
#     indices = np.where(rec['off_cells_samp'] == layer)
    
#     eta_r = rec['off_cells_deta'].take(indices[0])
#     phi_r = rec['off_cells_dphi'].take(indices[0])

#     eta = rec['off_cells_deta_digit'].take(indices[0])
#     phi = rec['off_cells_dphi_digit'].take(indices[0])
#     ene = rec['off_cells_e_norm'].take(indices[0])

#     indices_ =  (np.abs(eta) < n_eta) * (np.abs(phi) < n_phi) *(np.abs(eta_r) < r_eta) * (np.abs(phi_r) < r_phi)
        

#     eta_ = eta_r[indices_]
#     phi_ = phi_r[indices_]
#     ene_ = ene[indices_]
    
#     arr = np.array([eta_, phi_, ene_])
#     rec_new = np.core.records.fromarrays(
#         arr, names='x, y, z', formats='f8, f8, f8')
#     rec_new.sort(order=('x', 'y'))

#     if len(ene_) != npixels:
#         bad_img += 1
#     print ix, len(ene_), len(ene_) < npixels
    
#     if ix < 100 or len(ene_) != npixels:
#         plt.figure()
#         plt.scatter(
#             rec_new['x'], rec_new['y'], c=rec_new['z'], s=40,
#             marker='s', label= 'Number of cells = {0}'.format(len(eta_)))
#         plt.xlim(-0.4, 0.4)
#         plt.ylim(-0.4, 0.4)
#         plt.legend(loc='upper right', fontsize='small', numpoints=1)
#         plt.savefig('plots/grid_1p1n_S{0}_{1}.pdf'.format(layer, ix))
#         plt.clf()
#         plt.close()


# print bad_img, '/', len(rec_1p1n)
