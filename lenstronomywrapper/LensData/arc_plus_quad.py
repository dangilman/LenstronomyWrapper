import numpy as np
from copy import deepcopy
import itertools
from lenstronomywrapper.LensData.arc_plus_quad_util import LensedPointSource, SimulatedImage, DefaultDataSpecifics
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

class ArcPlusQuad(object):

    def __init__(self, x_image, y_image, magnifications, lensSystem, t_arrival=None, normed_magnifications=True,
                 data_kwargs={}):

        self.decimals_pos = 5
        self.decimals_mag = 6
        self.decimals_time = 1
        self.decimals_src = 8

        assert len(x_image) == 4
        assert len(x_image) == len(y_image)
        assert len(magnifications) == len(x_image)
        if t_arrival is not None:
            assert len(t_arrival) == len(x_image)
        self.x, self.y, self.m = np.array(x_image), np.array(y_image), np.array(magnifications)
        self.t_arrival = np.array(t_arrival)
        if t_arrival is not None:
            self.relative_arrival_times = self.t_arrival[1:] - self.t_arrival[0]

        if normed_magnifications:
            rescale_mag = 10
        else:
            rescale_mag = 1

        data_settings = DefaultDataSpecifics(**data_kwargs)
        self.kwargs_psf, args_data = data_settings()
        kwargs_data = sim_util.data_configure_simple(*args_data)
        data_class = ImageData(**kwargs_data)
        self.psf_class = PSF(**self.kwargs_psf)

        self.image_sim = SimulatedImage(x_image, y_image, rescale_mag * magnifications, lensSystem,
                                   data_class, self.psf_class, data_settings)

        kwargs_data['image_data'] = self.image_sim.get_image()
        data_class.update_data(kwargs_data['image_data'])

        self.kwargs_data = kwargs_data
        self.data_class = data_class
        self.point_source = self.image_sim.point_source

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

    def flux_ratios(self, index=0):

        ref_flux = self.m[index]
        ratios = []
        for i, mi in self.m:
            if i==index:
                continue
            ratios.append(mi/ref_flux)

        return np.array(ratios)

    def _sort_other(self, data):

        data_copy = deepcopy(data)
        data_copy.sort_by_pos(self.x, self.y)
        return data_copy

    def flux_anomaly(self, other_data=None, index=0, sum_in_quad=False):

        data_copy = self._sort_other(other_data)
        other_ratios = data_copy.flux_ratios(index)
        ratios = self.flux_ratios(index)

        if sum_in_quad:

            return np.sqrt((other_ratios - ratios) ** 2)

        else:

            return other_ratios - ratios


