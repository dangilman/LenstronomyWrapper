from lenstronomywrapper.Sampler.probability_distributions import *

class PriorDistribution(object):

    def __init__(self, distribution_type, args):

        if distribution_type == 'Uniform':
            self._func = Uniform(**args)

        elif distribution_type == 'Gaussian':
            self._func = Gaussian(**args)

        elif distribution_type == 'CustomPDF':
            self._func = CustomPDF(**args)

        else:
            raise Exception('distribution_type '+str(distribution_type)+' not recognized.')

    def __call__(self):

        return self._func()
