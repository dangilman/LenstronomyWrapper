import numpy as np
from scipy.interpolate import interp1d

n_decimals = 6

class Uniform(object):

    def __init__(self, low, high, positive_definite=False):

        self._low, self._high = low, high
        self._positive_definite = positive_definite

    def __call__(self):

        value = np.random.uniform(self._low, self._high)
        out = np.round(value, n_decimals)
        if self._positive_definite:
            out = abs(out)
        return out

class Gaussian(object):

    def __init__(self, mean, sigma, positive_definite=False):

        self._mean, self._sigma = mean, sigma

        self._positive_definite = positive_definite

    def __call__(self):
        value = np.random.normal(self._mean, self._sigma)
        out = np.round(value, n_decimals)
        if self._positive_definite:
            out = abs(out)
        return out

class CustomPDF(object):

    def __init__(self, parameter_values, probabilities, positive_definite=False):

        sorted = np.argsort(parameter_values)
        parameter_values = np.array(parameter_values)
        probabilities = np.array(probabilities)
        parameter_values_sorted = parameter_values[sorted]
        pdf = probabilities[sorted]
        norm = np.max(pdf)

        self.parameter_values, self.pdf = parameter_values_sorted, pdf / norm
        self._positive_definite = positive_definite

    def __call__(self):

        value = self._rejection_sampling()
        out = np.round(value, n_decimals)
        if self._positive_definite:
            out = abs(out)
        return out

    def _rejection_sampling(self):

        param_min, param_max = self.parameter_values[0], self.parameter_values[-1]

        pz = interp1d(self.parameter_values, self.pdf)

        while True:
            zprop = np.random.uniform(param_min, param_max)
            urand = np.random.uniform(0, 1)

            if pz(zprop) > urand:
                sample = zprop
                break
        return sample
