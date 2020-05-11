import numpy as np
from lenstronomywrapper.Utilities.data_util import image_separation_vectors_quad
from copy import deepcopy
import matplotlib.pyplot as plt
from lenstronomywrapper.Utilities.lensing_util import RayShootingGrid
from lenstronomy.LightModel.light_model import LightModel
from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase
from lenstronomywrapper.Utilities.lensing_util import flux_at_edge

class Quasar(SourceBase):

    def __init__(self, kwargs_quasars,
                 grid_resolution=None, grid_rmax=None):

        self._kwargs_init = kwargs_quasars
        self._grid_resolution = grid_resolution
        self._grid_rmax = grid_rmax
        self._grid_rmax_scale = 1
        self._initialized = False

        super(Quasar, self).__init__(False, [], None, None, None)

    @property
    def light_model_list(self):
        return ['GAUSSIAN']

    @property
    def kwargs_light(self):
        return self._kwargs_quasar


    def _check_initialized(self, with_error=True):

        if self._initialized:
            return True
        else:
            if with_error:
                raise Exception('Must initialize quasar class before using it.')
            return False

    def setup(self, pc_per_arcsec_zsource):

        if self._check_initialized(with_error=False):
            return

        self._initialized = True

        source_size_pc = self._kwargs_init['source_fwhm_pc']

        if self._grid_rmax is None:
            grid_rmax = self._auto_grid_size(source_size_pc)
            self.grid_rmax = grid_rmax
        else:
            self.grid_rmax = self._grid_rmax

        if self._grid_resolution is None:
            grid_resolution = self._auto_grid_resolution(source_size_pc)
            self.grid_resolution = grid_resolution
        else:
            self.grid_resolution = self._grid_resolution

        self._kwargs_quasar = self._kwargs_transform(self._kwargs_init, pc_per_arcsec_zsource)

        self._sourcelight = LightModel(light_model_list=['GAUSSIAN'])

    def update_position(self, x, y):

        self._kwargs_quasar['center_x'] = x
        self._kwargs_quasar['center_y'] = y

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        self._check_initialized()

        try:
            beta_x, beta_y = lensmodel.ray_shooting(xgrid, ygrid, lensmodel_kwargs)
            surf_bright = self._sourcelight.surface_brightness(beta_x, beta_y, [self._kwargs_quasar])

        except:
            shape0 = xgrid.shape
            beta_x, beta_y = lensmodel.ray_shooting(xgrid.ravel(), ygrid.ravel(), lensmodel_kwargs)
            surf_bright = self._sourcelight.surface_brightness(beta_x, beta_y, [self._kwargs_quasar])
            surf_bright = surf_bright.reshape(shape0, shape0)

        return surf_bright

    def _kwargs_transform(self, kwargs, pc_per_arcsec_zsrc):

        newkw = deepcopy(kwargs)
        fwhm_arcsec = newkw['source_fwhm_pc'] / pc_per_arcsec_zsrc
        newkw['sigma'] = fwhm_arcsec/2.355
        del newkw['source_fwhm_pc']
        if 'amp' not in newkw.keys():
            newkw['amp'] = 1

        return newkw

    def _ray_shooting_setup(self, xpos, ypos, grid_rmax_scale=1.):

        (image_separations, relative_angles) = image_separation_vectors_quad(xpos, ypos)

        grids = []
        grid_rmax = grid_rmax_scale * self.grid_rmax
        for sep, theta in zip(image_separations, relative_angles):
            grids.append(RayShootingGrid(min(grid_rmax, 0.5 * sep), self.grid_resolution, rot=theta))

        xgrids, ygrids = self._get_grids(xpos, ypos, grids)

        return xgrids, ygrids

    def get_images(self, xpos, ypos, lensModel, kwargs_lens,
                   grid_rmax_scale=1.):

        self._check_initialized()

        xgrids, ygrids = self._ray_shooting_setup(xpos, ypos,
                                            grid_rmax_scale)

        images, mags = [], []

        for i in range(0, len(xpos)):
            surface_brightness_image = self.surface_brightness(xgrids[i].ravel(), ygrids[i].ravel(),
                                                               lensModel, kwargs_lens)

            n = int(np.sqrt(len(surface_brightness_image)))

            images.append(surface_brightness_image.reshape(n, n))

        return images

    def plot_images(self, xpos, ypos, lensModel, kwargs_lens, normed=True):

        images = self.get_images(xpos, ypos, lensModel, kwargs_lens,
                                 self._grid_rmax_scale)

        mags = []
        for img in images:

            mags.append(np.sum(img) * self.grid_resolution ** 2)

        mags = np.array(mags)

        if normed:
            mags *= max(mags) ** -1
        for i in range(0, len(xpos)):

            plt.imshow(images[i])
            plt.annotate('relative magnification '+str(np.round(mags[i], 3)), xy=(0.1, 0.85), color='w',
                         xycoords='axes fraction')
            plt.show()

    def _flux_from_images(self, images):

        mags = []

        for image in images:
            if flux_at_edge(image):
                return None, True
            mags.append(np.sum(image) * self.grid_resolution ** 2)

        return np.array(mags), False

    def magnification(self, xpos, ypos, lensModel,
                      kwargs_lens, normed=True,
                      retry_if_blended=0):

        self._check_initialized()

        assert retry_if_blended >= 0
        assert retry_if_blended < 3

        grid_rmax_scale = 1

        images = self.get_images(xpos, ypos, lensModel,
                                 kwargs_lens, grid_rmax_scale)
        flux, blended = self._flux_from_images(images)

        blended_counter = 0

        while blended:

            if blended_counter >= retry_if_blended:
                break

            blended_counter += 1
            grid_rmax_scale *= 1.5
            self._grid_rmax_scale = grid_rmax_scale
            images = self.get_images(xpos, ypos, lensModel,
                                     kwargs_lens, grid_rmax_scale)
            flux, blended = self._flux_from_images(images)

        if normed and blended is False:
            flux *= np.max(flux) ** -1

        return flux, blended

    def _get_grids(self, xpos, ypos, grids):

        xgrid, ygrid = [], []

        for i, (xi, yi) in enumerate(zip(xpos, ypos)):

            xg, yg = grids[i].grid_at_xy(xi, yi)

            xgrid.append(xg)
            ygrid.append(yg)

        return xgrid, ygrid

    def _auto_grid_resolution(self, min_source_size_parsec):

        grid_res_0 = 0.000004
        size_0 = 0.1
        power = 1
        grid_res = grid_res_0 * (min_source_size_parsec / size_0) ** power

        return grid_res

    def _auto_grid_size(self, max_source_size_parsec):

        # smaller sources seem to require a different model that larger sources
        grid_size_0 = 0.0002
        size_0 = 0.1
        power = 1.15

        if max_source_size_parsec > 5:

            grid_size = grid_size_0 * (max_source_size_parsec / size_0) ** power

        else:

            power = 1.2
            grid_size_0 = 0.0004 / 5
            size_0 = 0.1 / 5

            grid_size = grid_size_0 * (max_source_size_parsec / size_0) ** power


        return grid_size

