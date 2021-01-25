from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
import numpy as np
from lenstronomy.Data.psf import PSF

class BaseReducedHST(object):

    def __init__(self, z_lens, z_source, image_data, x_image, y_image, deflector_centroid,
                 ra_at_xy_0, dec_at_xy_0, background_rms, transform_pix2angle, exposure_time,
                 mask, kernel_point_source, psf_error_map, psf_symmetry, satellite_centroid_list=None,
                 rescale_mag=100, magnifications=None, relative_arrival_times=None,
                 time_delay_sigma=None, image_sigma=0):

        self.z_lens = z_lens
        self.z_source = z_source
        self.image_data = image_data
        self.x = x_image
        self.y = y_image
        self.deflector_centroid = deflector_centroid
        self.ra_at_xy_0, self.dec_at_xy_0 = ra_at_xy_0, dec_at_xy_0
        self.background_rms = background_rms
        self.transform_pix2angle = transform_pix2angle
        self.exposure_time = exposure_time
        self.likelihood_mask = mask

        self.noise_map = None

        self.kernel_point_source_init = kernel_point_source
        self.psf_error_map_init = psf_error_map
        self.psf_symmetry = psf_symmetry

        self.updated_psf_error_map, self.updated_psf = None, None

        if relative_arrival_times is not None:
            assert len(relative_arrival_times) == len(x_image) - 1
            assert len(time_delay_sigma) == len(relative_arrival_times)
        else:
            relative_arrival_times = np.array([0.] * (len(x_image) - 1))
            time_delay_sigma = np.array([1000] * (len(x_image) - 1))

        self.relative_arrival_times = relative_arrival_times
        self.time_delay_sigma = time_delay_sigma

        if magnifications is None:
            magnifications = np.array([1] * len(x_image))

        self.m = magnifications
        self.point_source = LensedQuasar(x_image, y_image, rescale_mag * magnifications)

        self.image_sigma = image_sigma

        self.psf_class = PSF(**self.kwargs_psf)

        self.satellite_centroid_list = satellite_centroid_list

    def get_lensed_image(self, mask=False):

        if mask:
            N = len(self.image_data)
            data = self.image_data * self.likelihood_mask.reshape(N, N)
        else:
            data = self.image_data
        return data

    def update_psf(self, new_kps, new_error_map):

        self.updated_psf, self.updated_psf_error_map = new_kps, new_error_map

    @property
    def kwargs_psf(self):

        psf_estimate, error_map = self.best_psf_estimate

        kwargs_psf = {'psf_type': 'PIXEL',
                      'kernel_point_source': psf_estimate,
                      'psf_error_map': error_map}

        return kwargs_psf

    @property
    def best_psf_estimate(self):

        if self.updated_psf is None:
            print('using PSF estimate from initial star construction')
            return self.kernel_point_source_init, self.psf_error_map_init
        else:
            print('using PSF estimate from lenstronomy iteration during fitting sequence')
            return self.updated_psf, self.updated_psf_error_map

    @property
    def kwargs_data(self):

        kwargs_data = {'image_data': self.image_data,
                       'background_rms': self.background_rms,
                       'noise_map': self.noise_map,
                       'exposure_time': self.exposure_time,
                       'ra_at_xy_0': self.ra_at_xy_0,
                       'dec_at_xy_0': self.dec_at_xy_0,
                       'transform_pix2angle': self.transform_pix2angle
                       }

        return kwargs_data
