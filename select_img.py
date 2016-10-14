import os
from copy import deepcopy
import numpy as np
from root_numpy import tree2array
from rootpy.io import root_open
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
import skimage.transform as sk
from tauperf.parallel import Worker, run_pool
from tauperf import print_progress

def interpolate_rbf(x, y, z, function='linear', rotate_pc=True):
    """
    """
#     print 'interpolate ..'
    xi, yi = np.meshgrid(
        np.linspace(x.min(), x.max(), 14),
        np.linspace(y.min(), y.max(), 14))
    from scipy import interpolate
    rbf = interpolate.Rbf(x, y, z, function=function)
    im = rbf(xi, yi)
    if rotate_pc:
        scat_mat, cent = make_xy_scatter_matrix(x, y, z)
        #         scat_mat, cent = make_xy_scatter_matrix(xi, yi, im)
        paxes, pvars = get_principle_axis(scat_mat)
        angle = np.arctan2(paxes[0, 0], paxes[0, 1])
        im = sk.rotate(
            im, np.rad2deg(angle), order=3)
    return im

def tau_image(rec, cal_layer=2, rotate_pc=True):
    """
    """
    indices = np.where(rec['off_cells_samp'] == cal_layer)
    if len(indices) == 0:
        return None
    eta = rec['off_cells_deta'].take(indices[0])
    phi = rec['off_cells_dphi'].take(indices[0])
    ene = rec['off_cells_e_norm'].take(indices[0])
    if len(ene) == 0:
        return None
    image = interpolate_rbf(eta, phi, ene, rotate_pc=rotate_pc)
    return image, eta, phi, ene

def make_xy_scatter_matrix(x, y, z, scat_pow=2, mean_pow=1):

    cell_values = z
    cell_x = x
    cell_y = y

    etot = np.sum((cell_values>0) * np.power(cell_values, mean_pow))
    if etot == 0:
        print 'Found a jet with no energy.  DYING!'
        sys.exit(1)

    x_1  = np.sum((cell_values>0) * np.power(cell_values, mean_pow) * cell_x) / etot
    y_1  = np.sum((cell_values>0) * np.power(cell_values, mean_pow) * cell_y) / etot
    x_2  = np.sum((cell_values>0) * np.power(cell_values, scat_pow) * np.square(cell_x -x_1))
    y_2  = np.sum((cell_values>0) * np.power(cell_values, scat_pow) * np.square(cell_y -y_1))
    xy   = np.sum((cell_values>0) * np.power(cell_values, scat_pow) * (cell_x - x_1) * (cell_y -y_1))

    ScatM = np.array([[x_2, xy], [xy, y_2]])
    MeanV = np.array([x_1, y_1])

    return ScatM, MeanV

def get_principle_axis(mat):

    if mat.shape != (2,2):
        print "ERROR: getPrincipleAxes(theMat), theMat size is not 2x2. DYING!"
        sys.exit(1)

    las, lav = np.linalg.eigh(mat)
    return -1 * lav[::-1], las[::-1]



def dphi(phi_1, phi_2):
    d_phi = phi_1 - phi_2
    if (d_phi >= math.pi):
        return 2.0 * math.pi - d_phi
    if (d_phi < -1.0 * math.pi):
        return 2.0 * math.pi + d_phi
    return d_phi


data_dir = 'data_test'
rfile = root_open(os.path.join(
        os.getenv('DATA_AREA'), 
        'tauid_ntuples', 'output.root'))


tree = rfile['tau']
rec_1p1n = tree2array(
    tree, selection='true_nprongs==1 && true_npi0s == 1 && abs(off_eta) < 1.1').view(np.recarray)


print 'process 1p1n:', len(rec_1p1n)

for ix in xrange(len(rec_1p1n)):
    if ix > 100:
        break
    rec = rec_1p1n[ix]
    indices = np.where(rec['off_cells_samp'] == 2)
    
    eta = rec['off_cells_deta'].take(indices[0])
    phi = rec['off_cells_dphi'].take(indices[0])
    ene = rec['off_cells_e_norm'].take(indices[0])
    
    indices_ = (np.abs(eta) < 0.2) * (np.abs(phi) < 0.2)
    
    eta_ = eta[indices_]
    phi_ = phi[indices_]
    ene_ = ene[indices_]
    
    arr = np.array([eta_, phi_, ene_])
    rec_new = np.core.records.fromarrays(
        arr, names='x, y, z', formats='f8, f8, f8')
    rec_new.sort(order=('x', 'y'))


    print ix, len(ene_), len(ene_) < 256
    plt.figure()
    plt.scatter(
        rec_new['x'], rec_new['y'], c=rec_new['z'], 
        marker='s', label= 'Number of cells = {0}'.format(len(eta_)))
    plt.xlim(-0.4, 0.4)
    plt.ylim(-0.4, 0.4)
    plt.legend(loc='upper right', fontsize='small', numpoints=1)
    plt.savefig('plots/grid_1p1n_{0}.pdf'.format(ix))
    plt.clf()
    plt.close()

