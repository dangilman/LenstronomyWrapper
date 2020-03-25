import numpy as np
import itertools
from lenstronomywrapper.LensData.settings import DefaultDataSpecifics
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel

class ArcPlusQuad(object):

    def __init__(self, x_image, y_image, magnifications, lensSystem, relative_arrival_times=None, time_delay_sigma=None, image_sigma=None,
                 normed_magnifications=True,
                 data_kwargs={}, noiseless=True, no_bkg=True):

        assert len(x_image) == 4
        assert len(x_image) == len(y_image)
        assert len(magnifications) == len(x_image)

        if image_sigma is None:
            image_sigma = [0.0001]*len(x_image)
        self.image_sigma = image_sigma

        if relative_arrival_times is not None:
            assert len(relative_arrival_times) == len(x_image) - 1
            assert len(time_delay_sigma) == len(relative_arrival_times)

        self.x = np.array(x_image)
        self.y = np.array(y_image)
        self.m = np.array(magnifications)

        self.time_delay_sigma = time_delay_sigma
        self.relative_arrival_times = relative_arrival_times

        if normed_magnifications:
            rescale_mag = 25
        else:
            rescale_mag = 1

        data_settings = DefaultDataSpecifics(**data_kwargs)
        self.kwargs_psf, args_data = data_settings()
        kwargs_data = sim_util.data_configure_simple(*args_data)
        data_class = ImageData(**kwargs_data)
        self.psf_class = PSF(**self.kwargs_psf)

        self.point_source = LensedQuasar(x_image, y_image, rescale_mag*magnifications)

        self._image_sim = _LensData(self.point_source, lensSystem,
                                  data_class, self.psf_class, data_settings, noiseless, no_bkg)

        kwargs_data['image_data'] = self._image_sim._get_image()
        data_class.update_data(kwargs_data['image_data'])

        self.kwargs_data = kwargs_data
        self.data_class = data_class

    def get_lensed_image(self):

        return self._image_sim._get_image()

    def flux_ratios(self, index=0):

        ref_flux = self.m[index]
        ratios = []
        for i, mi in self.m:
            if i==index:
                continue
            ratios.append(mi/ref_flux)

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


class _LensData(object):

    def __init__(self, point_source, lensSystem, data_class, psf_class, data_settings,
                 noiseless, no_bkg):

        self._point_source = point_source
        lens_model, self._kwargs_lensmodel = lensSystem.get_lensmodel()
        lens_light_model, self._kwargs_lens_light = lensSystem.get_lens_light()
        source_model, self._kwargs_source_light = lensSystem.get_source_light()

        self._kwargs_ps = self._point_source.kwargs_ps

        self.imageModel = ImageModel(data_class, psf_class, lens_model, source_model,
                                     lens_light_model, self._point_source.point_source_class,
                                     kwargs_numerics=data_settings.kwargs_numerics)

        self._datasettings = data_settings
        self._noiseless = noiseless
        self._nobkg = no_bkg

    def _get_image(self):

        image = self.imageModel.image(self._kwargs_lensmodel, self._kwargs_source_light, self._kwargs_lens_light,
                                      self._kwargs_ps)

        if self._noiseless:
            poisson = 0
        else:
            poisson = image_util.add_poisson(image, exp_time=self._datasettings.exp_time)

        if self._nobkg:
            bkg = 0
        else:
            bkg = image_util.add_background(image, sigma_bkd=self._datasettings.background_rms)

        return image + bkg + poisson
