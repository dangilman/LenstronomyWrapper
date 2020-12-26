import numpy as np
from lenstronomywrapper.Utilities.data_util import image_separation_vectors_quad
from copy import deepcopy
import matplotlib.pyplot as plt
from lenstronomywrapper.Utilities.lensing_util import RayShootingGrid, AdaptiveGrid
from lenstronomy.LightModel.light_model import LightModel
from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase
from lenstronomywrapper.Utilities.lensing_util import flux_at_edge
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar

class DoubleGaussian(SourceBase):

    def __init__(self, kwargs_quasar,
                 grid_resolution=None, grid_rmax=None):

        """
        kwargs_quasar contains the keyword arguments: center_x, center_y, source_fwhm_pc, dx, dy, size_scale, amp_scale

        the first gaussian component is set by center_x, center_y, source_fwhm_pc, and the second is located at
        center_x + dx, center_y + dy, and the size is source_fwhm_pc * size_scale. The main source position, i.e. what
        is saved as source_centroid_x/y, corresponds to the first component at (center_x, center_y). The relative brightness
        of the second source to the first is set by amp_scale
        """
        self._kwargs_init = kwargs_quasar
        self._grid_resolution = grid_resolution
        self._grid_rmax = grid_rmax
        self._grid_rmax_scale = 1
        self._initialized = False

        self._kwargs_init = kwargs_quasar
        center_x, center_y = kwargs_quasar['center_x'], kwargs_quasar['center_y']
        source_fwhm_pc = kwargs_quasar['source_fwhm_pc']
        self._comp1 = Quasar({'center_x': center_x, 'center_y': center_y, 'source_fwhm_pc': source_fwhm_pc})

        center_x, center_y = kwargs_quasar['dx'] + center_x, kwargs_quasar['dy'] + center_y
        source_fwhm_pc = source_fwhm_pc * kwargs_quasar['size_scale']

        self._comp2 = Quasar({'center_x': center_x, 'center_y': center_y, 'source_fwhm_pc': source_fwhm_pc})

        super(DoubleGaussian, self).__init__(False, [], None, None, None)

    @property
    def light_model_list(self):
        return ['GAUSSIAN'] * 2

    @property
    def kwargs_light(self):

        return self._kwargs_quasar

    def _check_initialized(self, with_error=True):

        self._comp1._check_initialized(with_error)
        self._comp2._check_initialized(with_error)

    def setup(self, pc_per_arcsec_zsource=None):

        if self._check_initialized(with_error=False):
            return

        self._initialized = True

        if not hasattr(self, '_pc_per_arcsec_zsource'):
            assert pc_per_arcsec_zsource is not None
            self._pc_per_arcsec_zsource = pc_per_arcsec_zsource

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

        self._comp1.setup(pc_per_arcsec_zsource)

        self._comp2.setup(pc_per_arcsec_zsource)

        self._kwargs_quasar_1 = self._comp1._kwargs_quasar

        self._kwargs_quasar_2 = self._comp1._kwargs_quasar

        self._sourcelight = LightModel(light_model_list=['GAUSSIAN', 'GAUSSIAN'])

    def update_position(self, x, y):

        self._comp1.update_position(x, y)

        self._comp2.update_position(x + self._kwargs_init['dx'],
                                    y + self._kwargs_init['dy'])

    def surface_brightness_from_coords(self, beta_x, beta_y):

        self._check_initialized()

        shape0 = beta_x.shape

        surf_bright_1 = self._comp1.surface_brightness_from_coords(beta_x, beta_y)
        surf_bright_2 = self._comp2.surface_brightness_from_coords(beta_x, beta_y)

        return (surf_bright_1 + self._kwargs_init['amp_scale']*surf_bright_2).reshape(shape0)

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        self._check_initialized()

        try:
            beta_x, beta_y = lensmodel.ray_shooting(xgrid, ygrid, lensmodel_kwargs)
            surf_bright_1 = self._comp1.surface_brightness_from_coords(beta_x, beta_y)
            surf_bright_2 = self._comp2.surface_brightness_from_coords(beta_x, beta_y)
            surf_bright = surf_bright_1 + self._kwargs_init['amp_scale']*surf_bright_2
        except:
            shape0 = xgrid.shape
            beta_x, beta_y = lensmodel.ray_shooting(xgrid.ravel(), ygrid.ravel(), lensmodel_kwargs)
            surf_bright_1 = self._comp1.surface_brightness_from_coords(beta_x, beta_y)
            surf_bright_2 = self._comp2.surface_brightness_from_coords(beta_x, beta_y)
            surf_bright = surf_bright_1 + self._kwargs_init['amp_scale'] * surf_bright_2
            surf_bright = surf_bright.reshape(shape0, shape0)

        return surf_bright

    @staticmethod
    def _kwargs_transform(kwargs, pc_per_arcsec_zsrc):

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
            grids.append(RayShootingGrid(min(grid_rmax, 0.5 * sep), self.grid_resolution))

        xgrids, ygrids = self._get_grids(xpos, ypos, grids)

        return xgrids, ygrids

    def _ray_shooting_setup_adaptive(self, xpos, ypos, relative_angles):

        (image_separations, _) = image_separation_vectors_quad(xpos, ypos)

        grids = []
        grid_rmax = 2.5 * self.grid_rmax

        for sep, theta, xi, yi in zip(image_separations, relative_angles, xpos, ypos):

            end_rmax = min(grid_rmax, 0.5 * sep)

            new_grid = AdaptiveGrid(end_rmax, self.grid_resolution, theta,
                                    xi, yi)
            grids.append(new_grid)

        return grids

    def _iterate_adaptive(self, grid, r_min, r_max,
                          lensModel, kwargs_lens):

        xcoords, ycoords, inds = grid.get_coordinates(r_min, r_max)
        flux_in_pixels = self.surface_brightness(xcoords, ycoords, lensModel, kwargs_lens)
        grid.set_flux_in_pixels(inds, flux_in_pixels)
        magnification_current = np.sum(grid.flux_values) * grid.grid_res ** 2

        return magnification_current

    def magnification_adaptive(self, xpos, ypos, lensModel, kwargs_lens, normed, tol=0.005,
                               verbose=False, enforce_unblended=False, relative_angles=None):

        def _converged(dm):

            if dm < tol:
                return True
            else:
                return False

        if relative_angles is None:
            relative_angles = [0.] * len(xpos)

        grids = self._ray_shooting_setup_adaptive(xpos, ypos, relative_angles)

        self._adaptive_grids = grids

        step_factor = 0.1

        mags = []
        #print('computing magnification adaptive.... ')
        for i, grid in enumerate(grids):

            #print('computing image '+str(i))
            r_start = 0.05 * grid.rmax
            step_size = step_factor * grid.rmax

            #print('magnification: ', 0)
            magnification_last = self._iterate_adaptive(
                grid, 0., r_start, lensModel, kwargs_lens)
            converged = False
            r_min = r_start
            r_max = r_min + step_size

            if verbose:
                print('magnification: ', magnification_last)

            while converged is False:

                magnification_new = self._iterate_adaptive(
                    grid, r_min, r_max, lensModel, kwargs_lens)
                delta = 1 - magnification_last / magnification_new
                converged = _converged(delta)

                if verbose:
                    print('magnification: ', magnification_new)

                if r_max > grid.rmax:
                    break

                magnification_last = magnification_new
                r_min += step_size
                r_max += step_size

            if flux_at_edge(grid.image) and enforce_unblended:
                return None, True

            mags.append(magnification_new)

        fluxes = np.array(mags)

        if normed:
            fluxes *= np.max(fluxes) ** -1

        return fluxes, False

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

    def plot_images(self, xpos, ypos, lensModel, kwargs_lens, normed=True, adaptive=False):

        if adaptive:

            images = [grid.image for grid in self._adaptive_grids]

        else:
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

    def _flux_from_images(self, images, enforce_unblended):

        mags = []

        blended = False

        for image in images:

            if flux_at_edge(image):
                blended = True

            if blended and enforce_unblended:
                return None, True

            mags.append(np.sum(image) * self.grid_resolution ** 2)

        return np.array(mags), blended

    def magnification(self, xpos, ypos, lensModel,
                      kwargs_lens, normed=True, retry_if_blended=0,
                      enforce_unblended=False, adaptive=False, verbose=False):

        self._check_initialized()

        if adaptive:

            return self.magnification_adaptive(xpos, ypos, lensModel, kwargs_lens, normed,
                                               verbose=verbose, enforce_unblended=enforce_unblended)

        if enforce_unblended:

            return self._magnification_enforce_unblended(xpos, ypos, lensModel,
                      kwargs_lens, normed, retry_if_blended)

        else:

            return self._magnification(xpos, ypos, lensModel, kwargs_lens, normed)

    def _magnification(self, xpos, ypos, lensModel,
                      kwargs_lens, normed):

        images = self.get_images(xpos, ypos, lensModel,
                                 kwargs_lens, self._grid_rmax_scale)

        flux, blended = self._flux_from_images(images, False)

        if normed:
            flux *= np.max(flux) ** -1

        return flux, blended

    def _magnification_enforce_unblended(self, xpos, ypos, lensModel,
                      kwargs_lens, normed=True,
                      retry_if_blended=0,
                      enforce_unblended=True):

        self._check_initialized()

        if enforce_unblended:
            assert retry_if_blended >= 0
            assert retry_if_blended < 3

        grid_rmax_scale = 1

        blended_counter = 0

        while blended_counter <= retry_if_blended:

            self._grid_rmax_scale = grid_rmax_scale

            images = self.get_images(xpos, ypos, lensModel,
                                     kwargs_lens, grid_rmax_scale)

            flux, blended = self._flux_from_images(images, enforce_unblended)

            if blended is False:
                if normed:
                    flux *= np.max(flux) ** -1
                return flux, blended

            grid_rmax_scale = grid_rmax_scale + 1.5 * blended_counter
            blended_counter += 1

        else:

            return None, True

    def _get_grids(self, xpos, ypos, grids):

        xgrid, ygrid = [], []

        for i, (xi, yi) in enumerate(zip(xpos, ypos)):

            xg, yg = grids[i].grid_at_xy(xi, yi)

            xgrid.append(xg)
            ygrid.append(yg)

        return xgrid, ygrid

    def _auto_grid_resolution(self, min_source_size_parsec):

        grid_res_0 = 0.0000035
        size_0 = 0.1
        power = 1
        grid_res = grid_res_0 * (min_source_size_parsec / size_0) ** power

        return grid_res

    def _auto_grid_size(self, max_source_size_parsec):

        # smaller sources seem to require a different model that larger sources
        grid_size_0 = 0.0003
        size_0 = 0.1
        power = 1.15

        if max_source_size_parsec > 5:

            grid_size = grid_size_0 * (max_source_size_parsec / size_0) ** power

        else:

            power = 1.2
            grid_size_0 = 0.0006 / 5
            size_0 = 0.1 / 5

            grid_size = grid_size_0 * (max_source_size_parsec / size_0) ** power

        return grid_size

