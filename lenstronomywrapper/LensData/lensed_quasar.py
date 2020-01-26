import numpy as np
from lenstronomy.PointSource.point_source import PointSource
import itertools
from copy import deepcopy

class LensedQuasar(object):

    def __init__(self, x_image, y_image, mag, t_arrival=None):

        self.x, self.y = x_image, y_image
        self.m = mag
        self._nimg = len(x_image)
        self.t_arrival = t_arrival

        if t_arrival is not None:
            self.relative_arrival_times = self.t_arrival[1:] - self.t_arrival[0]

        if self._nimg == 4:
            pass
        else:
            raise Exception(str(self._nimg)+' lenses not yet incorporated.')

        point_amp = mag * 200  # multiply by intrinsic quasar brightness (in counts/s)

        self.kwargs_ps = [{'ra_image': self.x, 'dec_image': self.y,
                           'point_amp': point_amp}]  # quasar point source position in the source plane and intrinsic brightness
        self.point_source_list = ['LENSED_POSITION']

        self.point_source_class = PointSource(point_source_type_list=self.point_source_list,
                                              fixed_magnification_list=[False])

    def update_kwargs_ps(self, new_kwargs):

        self.kwargs_ps = new_kwargs

    def flux_ratios(self, index=0):

        ref_flux = self.m[index]
        ratios = []
        for i, mi in enumerate(self.m):
            if i == index:
                continue
            ratios.append(mi / ref_flux)

        return np.array(ratios)

    def sort_by_pos(self, x, y):

        x_self = np.array(list(itertools.permutations(self.x)))
        y_self = np.array(list(itertools.permutations(self.y)))

        indexes = [0, 1, 2, 3]
        index_iterations = list(itertools.permutations(indexes))
        delta_r = []

        for i in range(0, int(len(x_self))):
            dr = 0
            for j in range(0, int(len(x_self[0]))):
                dr += (x_self[i][j] - x[j]) ** 2 + (y_self[i][j] - y[j]) ** 2

            delta_r.append(dr ** .5)

        min_indexes = np.array(index_iterations[np.argmin(delta_r)])
        self.x = self.x[min_indexes]
        self.y = self.y[min_indexes]
        self.m = self.m[min_indexes]

        if self.t_arrival is not None:
            self.t_arrival = self.t_arrival[min_indexes]
            self.relative_arrival_times = self.t_arrival[1:] - self.t_arrival[0]

    @property
    def prior(self):
        return []

    @property
    def fixed_models(self):
        return [{}]

    @property
    def param_init(self):
        return self.kwargs_ps

    @property
    def param_sigma(self):
        sigma = [{'ra_image': [0.005] * self._nimg, 'dec_image': [0.005] * self._nimg}]
        return sigma

    @property
    def param_lower(self):
        lower = [{'ra_image': -10 * np.ones_like(self.x), 'dec_image': -10 * np.ones_like(self.y)}]
        return lower

    @property
    def param_upper(self):
        lower = [{'ra_image': 10 * np.ones_like(self.x), 'dec_image': 10 * np.ones_like(self.y)}]
        return lower
