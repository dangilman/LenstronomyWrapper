import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from scipy.interpolate import interp1d

def interpolate_ray_paths(x_image, y_image, lens_system, include_substructure=False, realization=None):

    ray_angles_x = []
    ray_angles_y = []

    lens_model, kwargs_lens = lens_system.get_lensmodel(include_substructure, True, realization)
    zsource = lens_system.zsource

    for (xi, yi) in zip(x_image, y_image):
        x, y, redshifts, tz = lens_model.lens_model.ray_shooting_partial_steps(0., 0., xi, yi, 0, zsource,
                                                                               kwargs_lens)

        angle_x = [xi] + [x_comoving / tzi for x_comoving, tzi in zip(x[1:], tz[1:])]
        angle_y = [yi] + [y_comoving / tzi for y_comoving, tzi in zip(y[1:], tz[1:])]
        ray_angles_x.append(interp1d(redshifts, angle_x))
        ray_angles_y.append(interp1d(redshifts, angle_y))

    return ray_angles_x, ray_angles_y

def ddt_from_h(H, omega_matter, omega_matter_baryon, zlens, zsource):
    _astro = FlatLambdaCDM(H0=float(H), Om0=omega_matter, Ob0=omega_matter_baryon)
    lensCosmo = LensCosmo(zlens, zsource, _astro)
    Dd = lensCosmo.D_d
    Ds = lensCosmo.D_s
    Dds = lensCosmo.D_ds
    return (1 + zlens) * Dd * Ds / Dds

def solve_H0_from_Ddt(zlens, zsource, D_dt, astropy_instance_ref):

    omega_matter = astropy_instance_ref.Om0
    omega_matter_baryon = astropy_instance_ref.Ob0
    out = []

    def _func_to_min(h0, d):
        return (ddt_from_h(h0, omega_matter, omega_matter_baryon, zlens, zsource) - d) ** 2

    if isinstance(D_dt, list) or isinstance(D_dt, np.ndarray):

        for di in D_dt:
            result = minimize(_func_to_min, x0=73.3,
                              method='Nelder-Mead', args=di)['x'][0]

            out.append(result)
    else:
        out = minimize(_func_to_min, x0=73.3,
                          method='Nelder-Mead', args=D_dt)['x'][0]

    return out

class RayShootingGrid(object):

    def __init__(self, side_length, grid_res, rot):

        N = int(2*side_length*grid_res**-1)

        if N==0:
            raise Exception('cannot raytracing with no pixels!')

        self.x_grid_0, self.y_grid_0 = np.meshgrid(
            np.linspace(-side_length+grid_res, side_length-grid_res, N),
            np.linspace(-side_length+grid_res, side_length-grid_res, N))

        self.radius = side_length

        self._rot = rot

    @property
    def grid_at_xy_unshifted(self):
        return self.x_grid_0, self.y_grid_0

    def grid_at_xy(self, xloc, yloc):

        theta = self._rot

        cos_phi, sin_phi = np.cos(theta), np.sin(theta)

        gridx0, gridy0 = self.grid_at_xy_unshifted

        _xgrid, _ygrid = (cos_phi * gridx0 + sin_phi * gridy0), (-sin_phi * gridx0 + cos_phi * gridy0)
        xgrid, ygrid = _xgrid + xloc, _ygrid + yloc

        xgrid, ygrid = xgrid.ravel(), ygrid.ravel()

        return xgrid, ygrid
