import numpy as np

class LensMaps(object):

    def __init__(self, system, multi_plane=True):

        self.system = system
        self.lensModel, self.kwargs_lens = self.system.get_lensmodel(set_multiplane=multi_plane)

    def _get_grids(self, rmin_max, npix):

        x = np.linspace(-rmin_max, rmin_max, npix)
        y = np.linspace(rmin_max, -rmin_max, npix)
        xx, yy = np.meshgrid(x, y)
        return xx.ravel(), yy.ravel(), xx.shape

    def convergence(self, rmin_max, npix):

        xgrid, ygrid, shape0 = self._get_grids(rmin_max, npix)
        kappa = self.lensModel.kappa(xgrid, ygrid, self.kwargs_lens).reshape(shape0)

        return kappa

    def fermat_potential(self, rmin_max, npix, xref, yref):

        xgrid, ygrid, shape0 = self._get_grids(rmin_max, npix)
        potential = self.lensModel.fermat_potential(xgrid, ygrid, self.kwargs_lens).reshape(shape0)
        potential_ref = self.lensModel.fermat_potential(xref, yref, self.kwargs_lens)
        return potential - potential_ref

    def time_delay_surface(self, rmin_max, npix, xref, yref):

        xgrid, ygrid, shape0 = self._get_grids(rmin_max, npix)
        arrival_time = self.lensModel.arrival_time(xgrid, ygrid, self.kwargs_lens).reshape(shape0)
        arrival_time_ref = self.lensModel.arrival_time(xref, yref, self.kwargs_lens)
        return arrival_time - arrival_time_ref

    def geo_shapiro_delay(self, rmin_max, npix, xref, yref):

        xgrid, ygrid, shape0 = self._get_grids(rmin_max, npix)
        dt_geo, dt_grav = self.lensModel.lens_model.geo_shapiro_delay(xgrid, ygrid, self.kwargs_lens)
        dt_geo_ref, dt_grav_ref = self.lensModel.lens_model.geo_shapiro_delay(xref, yref, self.kwargs_lens)

        dtgeo = dt_geo - dt_geo_ref
        dtgrav = dt_grav - dt_grav_ref

        return dtgeo.reshape(shape0), dtgrav.reshape(shape0)

    def curl(self, rmin_max, npix):

        xgrid, ygrid, shape0 = self._get_grids(rmin_max, npix)
        curl = self.lensModel.curl(xgrid, ygrid, self.kwargs_lens)
        return curl


class ResidualLensMaps(object):

    def __init__(self, map1, map2):

        self.map1 = map1
        self.map2 = map2

    def convergence(self, rmin_max, npix, mean0=False):

        kappa1 = self.map1.convergence(rmin_max, npix)
        kappa2 = self.map2.convergence(rmin_max, npix)
        residual = kappa1 - kappa2
        if mean0:
            residual += -np.mean(residual)
        return residual

    def curl(self, rmin_max, npix, mean0=False):

        curl1 = self.map1.curl(rmin_max, npix)
        curl2 = self.map2.curl(rmin_max, npix)
        residual = curl1 - curl2
        if mean0:
            residual += -np.mean(residual)
        return residual

    def fermat_potential(self, rmin_max, npix, xref, yref):

        pot1 = self.map1.fermat_potential(rmin_max, npix)
        pot2 = self.map2.fermat_potential(rmin_max, npix)

        pot1_ref = self.map1.lensModel.fermat_potential(xref, yref, self.map1.kwargs_lens)
        pot2_ref = self.map2.lensModel.fermat_potential(xref, yref, self.map2.kwargs_lens)
        pot1 -= pot1_ref
        pot2 -= pot2_ref

        residual = pot1 - pot2

        return residual

    def time_delay_surface(self, rmin_max, npix, xref, yref, x_point_eval=[], y_point_eval=[]):

        arrival_time_1 = self.map1.time_delay_surface(rmin_max, npix, xref, yref)
        arrival_time_2 = self.map2.time_delay_surface(rmin_max, npix, xref, yref)

        return arrival_time_1 - arrival_time_2

    def time_delay_surface_geoshapiro(self, rmin_max, npix, xref, yref, x_point_eval=[], y_point_eval=[]):

        dtgeo1, dtgrav1 = self.map1.geo_shapiro_delay(rmin_max, npix, xref, yref)
        dtgeo2, dtgrav2 = self.map2.geo_shapiro_delay(rmin_max, npix, xref, yref)

        return dtgeo1 - dtgeo2, dtgrav1 - dtgrav2

