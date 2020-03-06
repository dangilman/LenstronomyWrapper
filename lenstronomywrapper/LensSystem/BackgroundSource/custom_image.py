from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase
from lenstronomy.LightModel.Profiles.interpolation import Interpol
import numpy as np
from copy import deepcopy

import scipy
from lenstronomy.Cosmo.background import Background
import lenstronomy.Util.image_util as image_util

def coordinate_rescale(d_initial, z_final):

    background = Background()
    cosmo = background.cosmo

    #kpc_per_arcsec_initial = 1/cosmo.arcsec_per_kpc_proper(z_initial).value
    #kpc_per_arcsec_final = 1/cosmo.arcsec_per_kpc_proper(z_final).value
    #scale = kpc_per_arcsec_final/kpc_per_arcsec_initial

    scale = cosmo.comoving_transverse_distance(z_final).value/d_initial

    return scale

def setup_image(raw_image, ncut=200, cut_x=None, cut_y=None):

    npix_y, npix_x = raw_image.shape[0], raw_image.shape[1]
    if cut_x is not None and cut_y is not None:
        ncutx = npix_x * cut_x
        ncuty = npix_y * cut_y
        raw_image = raw_image[ncuty:(npix_y - ncuty), ncutx:(npix_x-ncutx)]

    median = np.median(raw_image[:ncut, :ncut])
    raw_image -= median

    # resize the image to square size (add zeros at the edges of the non-square bits of the image)
    nx, ny = np.shape(raw_image)
    n_min = min(nx, ny)
    n_max = max(nx, ny)
    image = np.zeros((n_max, n_max))
    x_start = int((n_max - nx) / 2.)
    y_start = int((n_max - ny) / 2.)
    image[x_start:x_start + nx, y_start:y_start + ny] = raw_image

    return image

def resize_image(image, z_initial, z_final, resize_factor=10):

    numPix_large = int(len(image) / resize_factor)
    n_new = int((numPix_large - 1) * resize_factor)
    cut = image[0:n_new, 0:n_new]

    n_pix_convolve = 5
    cut = scipy.ndimage.filters.gaussian_filter(cut, n_pix_convolve, mode='nearest', truncate=6)

    image = image_util.re_size(cut, resize_factor)  # re-size image to lower resolution

    pad_factor = 1.5
    add_pixels = int(0.5*(np.round(image.shape[0] * pad_factor) - image.shape[0]))

    new_image = np.pad(image, add_pixels, mode='constant')

    scale = coordinate_rescale(z_initial, z_final)
    new_size = new_image.shape[0]/image.shape[0]

    return new_image, 1/(scale * new_size)

class CustomImage(SourceBase):

    def __init__(self, kwargs, concentric_with_source=None):

        self.interpol = Interpol()
        self.image = kwargs[0]['image']

        self._kwargs = kwargs
        source_x, source_y = kwargs[0]['center_x'], kwargs[0]['center_y']
        super(CustomImage, self).__init__(concentric_with_source, [], source_x, source_y)

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        source_light_instance = self.sourceLight

        try:
            beta_x, beta_y = lensmodel.ray_shooting(xgrid, ygrid, lensmodel_kwargs)
            surf_bright = source_light_instance.surface_brightness(beta_x, beta_y, self.kwargs_light)

        except:
            shape0 = xgrid.shape
            beta_x, beta_y = lensmodel.ray_shooting(xgrid.ravel(), ygrid.ravel(), lensmodel_kwargs)
            surf_bright = source_light_instance.surface_brightness(beta_x, beta_y, self.kwargs_light)
            surf_bright = surf_bright.reshape(shape0)

        return surf_bright

    @property
    def light_model_list(self):
        return ['INTERPOL']

    @property
    def kwargs_light(self):
        return self._kwargs


