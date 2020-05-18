import numpy as np
from lenstronomywrapper.Utilities.data_util import image_separation_vectors_quad
from copy import deepcopy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from lenstronomywrapper.LensSystem.BackgroundSource.source_base import SourceBase
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomywrapper.Utilities.lensing_util import flux_at_edge

class Torus(SourceBase):

    def __init__(self, kwargs_torus, grid_resolution=None, grid_rmax=None):

        inner_radius = kwargs_torus['r_inner_pc']
        outer_radius = kwargs_torus['r_outer_pc']

        self._outer_radius, self._inner_radius = outer_radius, inner_radius

        assert inner_radius < outer_radius

        xcenter, ycenter = kwargs_torus['center_x'], kwargs_torus['center_y']

        kwargs_gaussian_1 = {'center_x': xcenter, 'center_y': ycenter, 'source_fwhm_pc': inner_radius}
        kwargs_gaussian_2 = {'center_x': xcenter, 'center_y': ycenter, 'source_fwhm_pc': outer_radius}

        self._outer_gaussian = Quasar(kwargs_gaussian_2, grid_resolution, grid_rmax)
        self._inner_gaussian = Quasar(kwargs_gaussian_1, grid_resolution, grid_rmax)

        self._outer_amp_scale = kwargs_torus['amp_scale']

        self._kwargs_torus = {'center_x': xcenter, 'center_y': ycenter,
                              'r_inner_pc': kwargs_torus['r_inner_pc'],
                              'r_outer_pc': kwargs_torus['r_outer_pc']}

        super(Torus, self).__init__(False, [], None, None, None)

    @property
    def normalization(self):

        if not hasattr(self, '_norm'):

            rmax = 3 * self._outer_radius
            outer_sigma = self._outer_radius / 2.355
            inner_sigma = self._inner_radius / 2.355
            x = y = np.linspace(-rmax, rmax, 500)
            xx, yy = np.meshgrid(x, y)
            rr = np.sqrt(xx ** 2 + yy ** 2)
            inner = np.exp(-0.5 * rr ** 2 / inner_sigma ** 2)
            outer = np.exp(-0.5 * rr ** 2 / outer_sigma ** 2)

            sb_torus = self._outer_amp_scale * outer - inner
            inds0 = np.where(sb_torus < 0)
            sb_torus[inds0] = 0

            flux_torus = np.sum(sb_torus)
            flux_gaussian = np.sum(outer)

            self._norm = flux_gaussian/flux_torus

        return self._norm

    @property
    def torus_sb(self):

        rmax = 3 * self._outer_radius

        outer_sigma = self._outer_radius / 2.355
        inner_sigma = self._inner_radius / 2.355

        x = y = np.linspace(-rmax, rmax, 500)
        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx ** 2 + yy ** 2)
        inner = np.exp(-0.5 * rr ** 2 / inner_sigma ** 2)
        outer = np.exp(-0.5 * rr ** 2 / outer_sigma ** 2)

        sb = self._outer_amp_scale * outer - inner

        return self.normalization * sb, rr

    @property
    def half_light_radius(self):

        sb, rr = self.torus_sb
        #sb_base = deepcopy(sb)
        total_flux = np.sum(sb)
        flux = 0
        rstep = (self._outer_radius - self._inner_radius)/500
        r = rstep

        while flux < 0.5*total_flux:

            inds = np.where(rr < r)

            flux += np.sum(sb[inds])
            sb[inds] = 0

            r += rstep

        return r - rstep

    @property
    def grid_resolution(self):
        return self._outer_gaussian.grid_resolution

    @property
    def kwargs_light(self):
        return self._kwargs_torus

    def setup(self, pc_per_arcsec_zsource):

        self._inner_gaussian.setup(pc_per_arcsec_zsource)
        self._outer_gaussian.setup(pc_per_arcsec_zsource)

        self._inner_gaussian.grid_resolution = self._outer_gaussian.grid_resolution
        self._inner_gaussian.grid_rmax = self._outer_gaussian.grid_rmax

    def update_position(self, x, y):

        self._outer_gaussian._kwargs_quasar['center_x'] = x
        self._outer_gaussian._kwargs_quasar['center_y'] = y

        self._inner_gaussian._kwargs_quasar['center_x'] = x
        self._inner_gaussian._kwargs_quasar['center_y'] = y

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

    def plot_images(self, xpos, ypos, lensModel, kwargs_lens, normed=True):

        images = self.get_images(xpos, ypos, lensModel, kwargs_lens)
        mags, _ = self._flux_from_images(images, False)

        if normed:
            mags *= max(mags) ** -1
        for i in range(0, len(xpos)):
            n = int(np.sqrt(len(images[i])))
            print('npixels: ', n)
            plt.imshow(images[i].reshape(n, n));
            plt.annotate('relative magnification '+str(np.round(mags[i], 3)), xy=(0.1, 0.85), color='w',
                         xycoords='axes fraction')
            plt.show()

    def surface_brightness(self, xgrid, ygrid, lensmodel, lensmodel_kwargs):

        surfbright_1 = self._inner_gaussian.surface_brightness(xgrid, ygrid,
                                                               lensmodel, lensmodel_kwargs)
        surfbright_2 = self._outer_gaussian.surface_brightness(xgrid, ygrid,
                                                               lensmodel, lensmodel_kwargs)

        surfbright = self._outer_amp_scale * surfbright_2 - surfbright_1

        inds0 = np.where(surfbright < 0)

        surfbright[inds0] = 0

        return self.normalization * surfbright

    def get_images(self, xpos, ypos, lensModel, kwargs_lens,
                   grid_rmax_scale=1.):

        images_1 = self._inner_gaussian.get_images(xpos, ypos, lensModel, kwargs_lens)
        images_2 = self._outer_gaussian.get_images(xpos, ypos, lensModel, kwargs_lens)

        images = []
        for (img1, img2) in zip(images_1, images_2):
            diff = self._outer_amp_scale * img2 - img1
            inds0 = np.where(diff < 0)
            diff[inds0] = 0
            images.append(self.normalization * diff)

        return images

    def magnification(self, xpos, ypos, lensModel, kwargs_lens, normed=True, enforce_unblended=False,
                      **kwargs):

        images = self.get_images(xpos, ypos, lensModel, kwargs_lens)

        magnifications, blended = self._flux_from_images(images, enforce_unblended)

        return magnifications, blended
