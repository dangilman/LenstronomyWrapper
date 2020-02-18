import numpy as np
from lenstronomywrapper.Utilities.data_util import image_separation_vectors_quad
from copy import deepcopy
import matplotlib.pyplot as plt
from lenstronomywrapper.Utilities.lensing_util import RayShootingGrid
from lenstronomy.LightModel.light_model import LightModel
from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase

class Torus(SourceBase):

    def __init__(self, kwargs_torus, grid_resolution=None, grid_rmax=None):

        self._kwargs_init = kwargs_torus
        self._grid_resolution = grid_resolution
        self._grid_rmax = grid_rmax
        self._initialized = False

        super(Torus, self).__init__(False, [], None, None)

    @property
    def light_model_list(self):
        return ['TORUS']

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

    def setup(self, pc_per_arcsec_zsource):

        if self._check_initialized(with_error=False):
            return

        self._initialized = True

        source_size_pc = self._kwargs_init['source_outerradius_pc']

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

        self._sourcelight = LightModel(light_model_list=['TORUS'])

    def update_position(self, x, y):

        self._kwargs_quasar['center_x'] = x
        self._kwargs_quasar['center_y'] = y

    def _kwargs_transform(self, kwargs, pc_per_arcsec_zsrc):

        newkw = deepcopy(kwargs)
        radius_outer = newkw['source_outerradius_pc'] / pc_per_arcsec_zsrc
        radius_inner = newkw['source_innerradius_pc'] / pc_per_arcsec_zsrc

        del newkw['source_outerradius_pc']
        del newkw['source_innerradius_pc']

        newkw['inner_radius'] = radius_inner
        newkw['outer_radius'] = radius_outer

        if 'amp' not in newkw.keys():
            newkw['amp'] = 1

        return newkw

    def _ray_shooting_setup(self, xpos, ypos):

        (image_separations, relative_angles) = image_separation_vectors_quad(xpos, ypos)

        grids = []
        for sep, theta in zip(image_separations, relative_angles):
            grids.append(RayShootingGrid(min(self.grid_rmax, 0.5 * sep), self.grid_resolution, rot=theta))

        xgrids, ygrids = self._get_grids(xpos, ypos, grids)

        return xgrids, ygrids

    def plot_images(self, xpos, ypos, lensModel, kwargs_lens, normed=True):

        self._check_initialized()

        xgrids, ygrids = self._ray_shooting_setup(xpos, ypos)

        images, mags = [], []

        for i in range(0, len(xpos)):
            surface_brightness_image = self.surface_brightness(xgrids[i].ravel(), ygrids[i].ravel(),
                                                               lensModel, kwargs_lens)

            images.append(surface_brightness_image)
            mags.append(np.sum(surface_brightness_image) * self.grid_resolution ** 2)

        mags = np.array(mags)

        if normed:
            mags *= max(mags) ** -1
        for i in range(0, len(xpos)):
            n = int(np.sqrt(len(images[i])))
            print('npixels: ', n)
            plt.imshow(images[i].reshape(n, n));
            plt.annotate('relative magnification '+str(np.round(mags[i], 3)), xy=(0.1, 0.85), color='w',
                         xycoords='axes fraction')
            plt.show()

    def magnification(self, xpos, ypos, lensModel, kwargs_lens, normed=True):

        self._check_initialized()

        flux = np.zeros_like(xpos)
        xgrids, ygrids = self._ray_shooting_setup(xpos, ypos)

        for i in range(0,len(xpos)):
            surface_brightness_image = self.surface_brightness(xgrids[i].ravel(), ygrids[i].ravel(),
                                                               lensModel, kwargs_lens)
            flux[i] = np.sum(surface_brightness_image * self.grid_resolution ** 2)

        flux = np.array(flux)

        if normed:
            flux *= max(flux) ** -1
        return flux

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

        #if max_source_size_parsec > 4:
        grid_size_0 = 0.000175
        size_0 = 0.1
        power = 1.15
        grid_size = grid_size_0 * (max_source_size_parsec / size_0) ** power
        # else:
        #     grid_size_0 = 0.000175
        #     size_0 = 0.1
        #     power = 1.3
        #     grid_size = grid_size_0 * (max_source_size_parsec / size_0) ** power

        return grid_size
