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

    def convergence(self, rmin_max, npix, center_x=0, center_y=0):

        xgrid, ygrid, shape0 = self._get_grids(rmin_max, npix)
        kappa = self.lensModel.kappa(center_x+xgrid, center_y+ygrid, self.kwargs_lens).reshape(shape0)

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

    def arrival_time_delay_geo_shapiro(self, x, y, xref, yref):

        t1geo, t1shap = self.lensModel.lens_model.geo_shapiro_delay(x, y, self.kwargs_lens)
        t2geo, t2shap = self.lensModel.lens_model.geo_shapiro_delay(xref, yref, self.kwargs_lens)
        return t1geo - t2geo, t1shap - t2shap

    def arrival_time_delay(self, x, y, xref, yref):

        dtgeo, dtgrav = self.arrival_time_delay_geo_shapiro(x, y, xref, yref)
        return dtgeo + dtgrav

    def curl(self, rmin_max, npix):

        xgrid, ygrid, shape0 = self._get_grids(rmin_max, npix)
        curl = self.lensModel.curl(xgrid, ygrid, self.kwargs_lens)
        return curl


class ResidualLensMaps(object):

    def __init__(self, lensmodel1, lensmodel2, kwargs1, kwargs2):

        self.lensmodel1, self.kwargs1 = lensmodel1, kwargs1
        self.lensmodel2, self.kwargs2 = lensmodel2, kwargs2

    def _get_grids(self, rminmax, npix):

        x, y = np.linspace(-rminmax, rminmax, npix), np.linspace(rminmax, -rminmax, npix)
        xx, yy = np.meshgrid(x, y)
        shape0 = xx.shape
        return xx.ravel(), yy.ravel(), shape0

    def time_delay_surface(self, rminmax, npix, xref, yref):

        res_geo, res_shapiro = self.time_delay_surface_geoshapiro(rminmax, npix, xref, yref)
        return res_geo + res_shapiro

    def time_delay_surface_12(self, rminmax, npix, xref, yref):

        xx, yy, shape0 = self._get_grids(rminmax, npix)

        arrival_time_surface1_geo, arrival_time_surface1_grav = self.lensmodel1.lens_model. \
            geo_shapiro_delay(xx.ravel(), yy.ravel(), self.kwargs1)
        arrival_time_surface1_geo_ref, arrival_time_surface1_grav_ref = self.lensmodel1.lens_model. \
            geo_shapiro_delay(xref, yref, self.kwargs1)

        arrival_time_surface2_geo, arrival_time_surface2_grav = self.lensmodel2.lens_model. \
            geo_shapiro_delay(xx.ravel(), yy.ravel(), self.kwargs2)
        arrival_time_surface2_geo_ref, arrival_time_surface2_grav_ref = self.lensmodel2.lens_model. \
            geo_shapiro_delay(xref, yref, self.kwargs2)

        surface1_geo = arrival_time_surface1_geo - arrival_time_surface1_geo_ref
        surface1_grav = arrival_time_surface1_grav - arrival_time_surface1_grav_ref

        surface2_geo = arrival_time_surface2_geo - arrival_time_surface2_geo_ref
        surface2_grav = arrival_time_surface2_grav - arrival_time_surface2_grav_ref

        surface1 = surface1_geo + surface1_grav
        surface2 = surface2_geo + surface2_grav

        return surface1.reshape(shape0), surface2.reshape(shape0)

    def time_delay_surface_geoshapiro(self, rminmax, npix, xref, yref):

        xx, yy, shape0 = self._get_grids(rminmax, npix)

        arrival_time_surface1_geo, arrival_time_surface1_grav = self.lensmodel1.lens_model.\
            geo_shapiro_delay(xx.ravel(), yy.ravel(), self.kwargs1)
        arrival_time_surface1_geo_ref, arrival_time_surface1_grav_ref = self.lensmodel1.lens_model.\
            geo_shapiro_delay(xref, yref, self.kwargs1)

        arrival_time_surface2_geo, arrival_time_surface2_grav = self.lensmodel2.lens_model. \
            geo_shapiro_delay(xx.ravel(), yy.ravel(), self.kwargs2)
        arrival_time_surface2_geo_ref, arrival_time_surface2_grav_ref = self.lensmodel2.lens_model. \
            geo_shapiro_delay(xref, yref, self.kwargs2)

        surface1_geo = arrival_time_surface1_geo - arrival_time_surface1_geo_ref
        surface1_grav = arrival_time_surface1_grav - arrival_time_surface1_grav_ref

        surface2_geo = arrival_time_surface2_geo - arrival_time_surface2_geo_ref
        surface2_grav = arrival_time_surface2_grav - arrival_time_surface2_grav_ref

        residual_geo = surface1_geo - surface2_geo
        residual_grav = surface1_grav - surface2_grav

        return residual_geo.reshape(shape0), residual_grav.reshape(shape0)

    def convergence(self, rminmax, npix, mean0=False):

        xx, yy, shape0 = self._get_grids(rminmax, npix)

        kappa1 = self.lensmodel1.kappa(xx.ravel(), yy.ravel(), self.kwargs1)
        kappa2 = self.lensmodel2.kappa(xx.ravel(), yy.ravel(), self.kwargs2)
        residual = kappa1 - kappa2
        if mean0:
            residual -= np.mean(residual)
        return residual.reshape(shape0)
