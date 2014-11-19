import re

from . import log; log = log[__name__]
from . import samples
from .categories import CATEGORIES
from . import NTUPLE_PATH
VAR_PATTERN = re.compile('((?P<prefix>hlt|off|true)_)?(?P<var>[A-Za-z0-9_]+)(\*(?P<scale>\d+\.\d*))?$')


class Analysis(object):
    
    def __init__(self, ntuple_path=NTUPLE_PATH):

        self.tau = samples.Tau(
            ntuple_path=ntuple_path,
            name='tau', label='Real #tau_{had}',
            color='#00A3FF')

#         self.jet = samples.Jet(
#             name='jet', 
#             student='jetjet_JZ7W',
#             label='Fake #tau_{had}',
#             color='#00FF00')

        self.jet = samples.JZ(
            ntuple_path=ntuple_path,
            name='jet', 
            label='Fake #tau_{had}',
            color='#00FF00')
        self.jet.set_scales([1., 1., 1., 1.])

    def iter_categories(self, *definitions, **kwargs):
        names = kwargs.pop('names', None)
        for definition in definitions:
            for category in CATEGORIES[definition]:
                if names is not None and category.name not in names:
                    continue
                log.info("")
                log.info("=" * 40)
                log.info("%s category" % category.name)
                log.info("=" * 40)
                log.info("Cuts: %s" % self.tau.cuts(category))
                yield category

    def get_hist_samples_array(self, vars, prefix, category=None, cuts=None):
        """
        """
        field_hist_tau = self.tau.get_field_hist(vars, prefix)
        log.info(field_hist_tau)
        log.debug('Retrieve Tau histograms')
        field_hist_tau = self.tau.get_hist_array(field_hist_tau, category=category, cuts=cuts)
        field_hist_jet = self.jet.get_field_hist(vars, prefix)
        log.debug('Retrieve Jet histograms')
        field_hist_jet = self.jet.get_hist_array(field_hist_jet, category=category, cuts=cuts)
        hist_samples_array = {}
        for key in field_hist_tau:
            match = re.match(VAR_PATTERN, key)
            if match:
                hist_samples_array[match.group('var')] = {
                    'tau': field_hist_tau[key],
                    'jet': field_hist_jet[key]
                }
            else:
                log.warning('No pattern matching for {0}'.format(key))
        return hist_samples_array

    def get_hist_signal_array(self, vars, prefix1, prefix2, category=None, cuts=None):
        """
        """
        field_hist_tau_1 = self.tau.get_field_hist(vars, prefix1)
        field_hist_tau_2 = self.tau.get_field_hist(vars, prefix2)
        log.debug('Retrieve Tau histograms')
        field_hist_tau_1 = self.tau.get_hist_array(field_hist_tau_1, category=category, cuts=cuts)
        field_hist_tau_2 = self.tau.get_hist_array(field_hist_tau_2, category=category, cuts=cuts)
        log.info(field_hist_tau_1)
        log.info(field_hist_tau_2)
        
        hist_samples_array = {}
        for key in field_hist_tau_1:
            match = re.match(VAR_PATTERN, key)
            if match:
                field_hist_tau_1[key].title += ' ({0})'.format(prefix1)
                hist_samples_array[match.group('var')] = {prefix1: field_hist_tau_1[key]}
        for key in field_hist_tau_2:
            match = re.match(VAR_PATTERN, key)
            if match:
                field_hist_tau_2[key].title += ' ({0})'.format(prefix2)
                hist_samples_array[match.group('var')][prefix2] = field_hist_tau_2[key]
        return hist_samples_array