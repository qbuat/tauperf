import os
import numpy as np
from sklearn import model_selection
import tables
import math

from .. import print_progress
from . import log; log = log.getChild(__name__)
#from . import log; log = log[__name__]



class Image(tables.IsDescription):

    s1 = tables.Float64Col(shape=(4, 128), dflt=0.0)
    s2 = tables.Float64Col(shape=(32, 32), dflt=0.0)
    s3 = tables.Float64Col(shape=(32, 16), dflt=0.0)
    s4 = tables.Float64Col(shape=(16, 16), dflt=0.0)
    s5 = tables.Float64Col(shape=(16, 16), dflt=0.0)

#     s1 = tables.Float32Col(shape=(32, 128), dflt=0.0)
#     s2 = tables.Float32Col(shape=(32, 128), dflt=0.0)
#     s3 = tables.Float32Col(shape=(32, 128), dflt=0.0)
#     s4 = tables.Float32Col(shape=(32, 128), dflt=0.0)
#     s5 = tables.Float32Col(shape=(32, 128), dflt=0.0)

    tracks = tables.Float64Col(shape=(15, 4))
    e = tables.Float64Col()
    pt = tables.Float64Col()
    eta = tables.Float64Col()
    phi = tables.Float64Col()
    mu = tables.Float64Col()
    pantau = tables.Float64Col()
    truthmode = tables.Float64Col()
    true_pt = tables.Float64Col()
    true_eta = tables.Float64Col()
    true_phi = tables.Float64Col()
    true_m = tables.Float64Col()
    alpha_e = tables.Float64Col()



def tau_topo_image(evt, cal_layer=2, width=32, height=32):
    """
    """

    samp_ = np.array(evt.off_cells_samp)
    indices = np.where(samp_ == cal_layer)

    ene_ = np.array(evt.off_cells_e_norm).take(indices[0])
    eta_ = np.array(evt.off_cells_deta_digit).take(indices[0])
    phi_ = np.array(evt.off_cells_dphi_digit).take(indices[0])

    # build a rectangle of 0 to start
    # [[0 for j in range(width)] for i in range(height)]
    image = np.zeros((height, width))

    # loop over the recorded cells:
    # locate their position in the rectangle and replace the 0 by the energy value
    for eta, phi, ene in zip(eta_, phi_, ene_):
        eta_ind = int(eta + math.floor(width / 2))
        phi_ind = int(phi + math.floor(height / 2))
        if eta_ind < width  and eta_ind > 0 and phi_ind < height and phi_ind > 0:
            # print ene, 1e6 * ene, int(1e6 * ene)
            image[phi_ind][eta_ind] = ene
    image = np.asarray(image)

    return image

def tau_tracks_simple(evt):
    """
    Laser was here.
    """
    maxtracks = 15

    imp   = []
    deta  = []
    dphi  = []
    classes  = []


    indices = np.where(np.array(evt.off_tracks_deta) > -1000)

    rp     = np.array(evt.off_tracks_p)            .take(indices[0])
    rdeta  = np.array(evt.off_tracks_deta)           .take(indices[0])
    rdphi  = np.array(evt.off_tracks_dphi)           .take(indices[0])
    rclass = np.array(evt.off_tracks_class)          .take(indices[0])

    tau_ene = evt.off_e

    itrack = 0
    for (p, e, f, c) in zip(rp, rdeta, rdphi, rclass):
        itrack +=1
        if itrack == maxtracks:
            break
        imp.append(p / tau_ene)
        deta.append(e)
        dphi.append(f)
        classes.append(c)
        
    imp   += [0] * (maxtracks - len(imp))
    deta  += [0] * (maxtracks - len(deta))
    dphi  += [0] * (maxtracks - len(dphi))
    classes += [0] * (maxtracks - len(classes))

    tracks = zip(imp, deta, dphi, classes)
    tracks = np.asarray(tracks)

    return tracks

def process_taus(
    out_h5,
    tree,
    tree_number,
    nentries=None, 
    n_chunks=3,
    do_plot=False, 
    suffix='1p1n', 
    show_progress=True):
    '''
    process the records one by one and compute the image for each layer
    ----------
    returns a record array of the images + basic kinematics of the tau
    '''

    if nentries is None:
        nentries = tree.GetEntries()

    if not out_h5.root.__contains__('data'):
        out_h5.create_group('/', 'data', 'yup')
    group = out_h5.root.data

    filters = tables.Filters(complevel=1)

    table_name = 'table_{}'.format(tree_number)
    if not group.__contains__(table_name):
        log.info('Creating {} for {} taus'.format(table_name, suffix))
        out_h5.create_table(group, table_name, Image, filters=filters)

    table = getattr(out_h5.root.data, table_name)
    log.info('Filling table {0} for {1} taus with {2} entries '.format(table_name, suffix, nentries))
    for entry in xrange(nentries):
        if show_progress: 
            print
            print_progress(entry, nentries, prefix='Progress')
            print
        # protect against pathologic arrays
        try:
            tree.GetEntry(entry)
                # rec = records[index]
        except:
            log.warning('record array is broken')
            continue

        # get the image for each layer
        s1 = tau_topo_image(tree, cal_layer=1, width=Image.columns['s1'].shape[1], height=Image.columns['s1'].shape[0])
        s2 = tau_topo_image(tree, cal_layer=2, width=Image.columns['s2'].shape[1], height=Image.columns['s2'].shape[0])
        s3 = tau_topo_image(tree, cal_layer=3, width=Image.columns['s3'].shape[1], height=Image.columns['s3'].shape[0])
        s4 = tau_topo_image(tree, cal_layer=12,width=Image.columns['s4'].shape[1], height=Image.columns['s4'].shape[0])
        s5 = tau_topo_image(tree, cal_layer=13,width=Image.columns['s5'].shape[1], height=Image.columns['s5'].shape[0])

            # making all the images as (32 X 128)
#             s1_repeat = np.repeat(s1, 8, axis=0)
#             s2_repeat = np.repeat(s2, 4, axis=1)
#             s3_repeat = np.repeat(s3, 8, axis=1)
#             s4_repeat = np.repeat(s4, 2, axis=0)
#             s4_repeat = np.repeat(s4_repeat, 8, axis=1)
#             s5_repeat = np.repeat(s5, 2, axis=0)
#             s5_repeat = np.repeat(s5_repeat, 8, axis=1)

        table.row['s1'] = s1#_repeat
        table.row['s2'] = s2#_repeat
        table.row['s3'] = s3#_repeat
        table.row['s4'] = s4#_repeat
        table.row['s5'] = s5#_repeat
        table.row['tracks'] = tau_tracks_simple(tree)
        table.row['e'] = tree.off_e
        table.row['pt'] = tree.off_pt
        table.row['eta'] = tree.off_eta
        table.row['phi'] = tree.off_phi
        table.row['mu'] = tree.averageintpercrossing
        table.row['pantau'] = tree.off_decaymode
        table.row['truthmode'] = tree.true_decaymode
        table.row['true_pt'] = tree.true_pt
        table.row['true_eta'] = tree.true_eta
        table.row['true_phi'] = tree.true_phi
        table.row['true_m'] = tree.true_m
        table.row['alpha_e'] = tree.true_alpha_e
        table.row.append()

        if do_plot:
            import matplotlib as mpl; mpl.use('PS')
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            for i, im in enumerate([s1, s2, s3, s4, s5]):
                fig = plt.figure()
                plt.imshow(
                    im, extent=[-0.2, 0.2, -0.2, 0.2], cmap=plt.cm.Reds,
                    interpolation='nearest',
                    norm=LogNorm(0.0001, 1))
                plt.colorbar()
                plt.xlabel('eta')
                plt.ylabel('phi')
                plt.title('{0}: image {1} sampling {2}'.format(suffix, index, i + 1))
                fig.savefig('s{0}_tau_{1}.pdf'.format(i + 1, index))
                fig.clf()
                plt.close()

    # flush the table on disk
    table.flush()



