from rootpy.tree import Cut
from rootpy.plotting import Hist
from rootpy import asrootpy
import ROOT

# local imports
from .db import get_file
from .. import NTUPLE_PATH, DEFAULT_STUDENT, DEFAULT_TREE
from .. import log; log = log[__name__]


class Sample(object):

    def __init__(self, cuts=None,
                 ntuple_path=NTUPLE_PATH,
                 student=DEFAULT_STUDENT,
                 tree_name=DEFAULT_TREE,
                 name='Sample',
                 label='Sample',
                 **hist_decor):

        if cuts is None:
            self._cuts = Cut()
        else:
            self._cuts = cuts
        self.ntuple_path = ntuple_path
        self.student = student
        self.tree_name = tree_name
        self.name = name
        self.label = label
        self.hist_decor = hist_decor

        if 'fillstyle' not in hist_decor:
            self.hist_decor['fillstyle'] = 'solid'

    def decorate(self, name=None, label=None, **hist_decor):
        if name is not None:
            self.name = name
        if label is not None:
            self.label = label
        if hist_decor:
            self.hist_decor.update(hist_decor)
        return self

    def events(self, category=None, cuts=None):
        selection = Cut(self._cuts)
        if cuts is not None:
            selection &= cuts
        if category is not None:
            selection &= self.cuts(category)
        return self.draw_helper(Hist(1, 0.5, 1.5), '1', selection)

    def cuts(self, category=None, **kwargs):
        cuts = Cut(self._cuts)
        if category is not None:
            cuts &= category.get_cuts(**kwargs)
        return cuts


    def get_field_hist(self, vars, prefix=None):
        """
        """
        field_hist = {}
        for field, var_info in vars.items():
            nbins, xmin, xmax = var_info['bins'], var_info['range'][0], var_info['range'][1]
            exprs = []
            if 'prefix' in var_info:
                if not isinstance(var_info['prefix'], (list)):
                    var_info['prefix'] = [var_info['prefix']]
                from .jet import Jet
                if isinstance(self, Jet) and 'true' in  var_info['prefix']:
                     var_info['prefix'].remove('true')
                if prefix is None:
                    for p in var_info['prefix']:
                        var_name = p + '_' + var_info['name']
                        exprs.append(var_name)
                else:
                    var_name = prefix + '_' + var_info['name']
                    exprs.append(var_name)
            else:
                exprs.append(var_info['name'])
            if 'scale' in var_info:
                for i, expr in enumerate(exprs):
                    exprs[i] = '{0}*{1}'.format(expr, var_info['scale'])
            log.debug(exprs)
            for expr in exprs:
                field_hist[expr] = Hist(
                    nbins, xmin, xmax, type='D')
                
        return field_hist

    def draw_helper(self, hist_template, expr, selection): 
        """
        """
        rfile = get_file(self.ntuple_path, self.student)
        tree = rfile[self.tree_name]
        # use TTree Draw for now (limited to Nbins, Xmin, Xmax)
        binning = (
            hist_template.nbins(), 
            list(hist_template.xedges())[0], 
            list(hist_template.xedges())[-1])
        hist_template.Sumw2()
        root_string = '{0}>>{1}{2}'.format(
            expr, hist_template.name, binning)
        log.debug("Plotting {0} using selection: {1}".format(
                root_string, selection))
        log.debug('Draw {0} with \n selection: {1} ...'.format(root_string, selection))
        tree.Draw(root_string, selection)

        try:
            hist = asrootpy(ROOT.gPad.GetPrimitive(hist_template.name))
            return Hist(hist, title=self.label, **self.hist_decor)
        except:
            log.warning('{0}: unable to retrieve histogram for {1} with selection {2}'.format(
                    self.name, expr, selection))
            return Hist(binning[0], binning[1], binning[2], title=self.label, **self.hist_decor)

    def get_hist_array(self, field_hist_template, category=None, cuts=None):
        """
        """
        selection = self.cuts(category)
        if not cuts is None:
            seletion &= cuts
        field_hists = {}
        for key, hist in field_hist_template.items():
            field_hists[key] = self.draw_helper(hist, key, selection)
        return field_hists
