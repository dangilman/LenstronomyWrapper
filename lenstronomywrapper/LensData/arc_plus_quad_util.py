import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
import numpy as np

class LensedPointSource(object):

    def __init__(self, x_image, y_image, mag):

        self.x_image, self.y_image = x_image, y_image
        self._nimg = len(x_image)
        # perturb observed magnification due to e.g. micro-lensing
        # mag_pert = np.random.normal(mag, 0.5, len(mag))
        point_amp = mag * 100  # multiply by intrinsic quasar brightness (in counts/s)

        self.kwargs_ps = [{'ra_image': self.x_image, 'dec_image': self.y_image,
                           'point_amp': point_amp}]  # quasar point source position in the source plane and intrinsic brightness
        self.point_source_list = ['LENSED_POSITION']

        self.point_source_class = PointSource(point_source_type_list=self.point_source_list,
                                              fixed_magnification_list=[False])

    @property
    def fixed_models(self):
        return [{}]

    @property
    def param_init(self):
        return self.kwargs_ps

    @property
    def param_sigma(self):
        sigma = [{'ra_image': [0.05] * self._nimg, 'dec_image': [0.05] * self._nimg}]
        return sigma

    @property
    def param_lower(self):
        lower = [{'ra_image': -10 * np.ones_like(self.x_image), 'dec_image': -10 * np.ones_like(self.y_image)}]
        return lower

    @property
    def param_upper(self):
        lower = [{'ra_image': 10 * np.ones_like(self.x_image), 'dec_image': 10 * np.ones_like(self.y_image)}]
        return lower

class SimulatedImage(object):

    def __init__(self, x_image, y_image, magnifications, lensSystem, data_class, psf_class, data_settings, noiseless=True):

        self.point_source = LensedPointSource(x_image, y_image, magnifications)
        lens_model, self._kwargs_lensmodel = lensSystem.get_lensmodel()
        lens_light_model, self._kwargs_lens_light = lensSystem.get_lens_light()
        source_model, self._kwargs_source_light = lensSystem.get_source_light()

        self._kwargs_ps = self.point_source.kwargs_ps

        self.imageModel = ImageModel(data_class, psf_class, lens_model, source_model,
                                     lens_light_model, self.point_source.point_source_class,
                                     kwargs_numerics=data_settings.kwargs_numerics)

        self._datasettings = data_settings
        self._noiseless = noiseless

    def get_image(self):

        image = self.imageModel.image(self._kwargs_lensmodel, self._kwargs_source_light, self._kwargs_lens_light,
                                      self._kwargs_ps)

        if self._noiseless:
            poisson = 0
        else:
            poisson = image_util.add_poisson(image, exp_time=self._datasettings.exp_time)

        bkg = image_util.add_background(image, sigma_bkd=self._datasettings.background_rms)

        return image + bkg + poisson

class DefaultDataSpecifics(object):

    def __init__(self, background_rms=0.5, exp_time=100, numPix=80, deltaPix=0.04, fwhm=0.1):

        self.background_rms = background_rms
        self.exp_time = exp_time
        self.numPix = numPix
        self.deltaPix = deltaPix
        self.fwhm = fwhm
        self.kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    def __call__(self):
        out_psf = {'psf_type': 'GAUSSIAN', 'fwhm': self.fwhm, 'pixel_size': self.deltaPix, 'truncation': 5}
        out_data = (self.numPix, self.deltaPix, self.exp_time, self.background_rms)
        return out_psf, out_data
