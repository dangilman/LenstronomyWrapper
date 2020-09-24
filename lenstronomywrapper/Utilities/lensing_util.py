import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from scipy.interpolate import interp1d
from time import time

def flux_at_edge(image):

    assert np.shape(image)[0] == np.shape(image)[1]

    maxbright = np.max(image)
    edgebright = [image[0,:],image[-1,:],image[:,0],image[:,-1]]

    for edge in edgebright:
        if any(edge > maxbright * 0.2):
            return True
    else:
        return False

def interpolate_ray_path_center(x_center, y_center, source_x, source_y, lens_system,
                                include_substructure=False, realization=None):

    # load the lens model and keywords for ray tracing
    lens_model, kwargs_lens = lens_system.get_lensmodel(include_substructure, True, realization)
    zsource = lens_system.zsource

    # ray trace through the lens model
    x, y, redshifts, tz = lens_model.lens_model.ray_shooting_partial_steps(
        0., 0., x_center, y_center, 0, zsource, kwargs_lens)

    # compute the angular coordinate of the ray at each lens plane
    angle_x = [x_center] + [x_comoving / tzi for x_comoving, tzi in zip(x[1:], tz[1:])]
    angle_y = [y_center] + [y_comoving / tzi for y_comoving, tzi in zip(y[1:], tz[1:])]

    # replace the final angular coordinate with the source coordinate
    angle_x[-1] = source_x
    angle_y[-1] = source_y

    # interpolate
    ray_angles_x = interp1d(redshifts, angle_x)
    ray_angles_y = interp1d(redshifts, angle_y)

    return [ray_angles_x], [ray_angles_y]

def interpolate_ray_paths(x_image, y_image, lens_model, kwargs_lens, zsource,
                          terminate_at_source=False, source_x=None, source_y=None):

    """
    :param x_image: x coordinates to interpolate (arcsec)
    :param y_image: y coordinates to interpolate (arcsec)
    :param lens_model: instance of LensModel
    :param kwargs_lens: keyword arguments for lens model
    :param zsource: source redshift
    :param terminate_at_source: fix the final angular coordinate to the source coordinate
    :param source_x: source x coordinate (arcsec)
    :param source_y: source y coordinate (arcsec)
    :return: Instances of interp1d (scipy) that return the angular coordinate of a ray given a
    comoving distance
    """

    ray_angles_x = []
    ray_angles_y = []
    #print('coordinate: ', (x_image, y_image))
    for (xi, yi) in zip(x_image, y_image):

        angle_x, angle_y, tz = ray_angles(xi, yi, lens_model, kwargs_lens, zsource)

        if terminate_at_source:
            angle_x[-1] = source_x
            angle_y[-1] = source_y

        ray_angles_x.append(interp1d(tz, angle_x))
        ray_angles_y.append(interp1d(tz, angle_y))

    return ray_angles_x, ray_angles_y

def ray_angles(alpha_x, alpha_y, lens_model, kwargs_lens, zsource):

    redshift_list = lens_model.redshift_list + [zsource]

    x_angle_list, y_angle_list, tz = [alpha_x], [alpha_y], [0.]

    cosmo_calc = lens_model.lens_model._multi_plane_base._cosmo_bkg.T_xy

    x0, y0 = 0., 0.
    zstart = 0.

    for zi in np.unique(redshift_list):
        x0, y0, alpha_x, alpha_y = lens_model.lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, zstart, zi,
                                                                               kwargs_lens)
        d = cosmo_calc(0., zi)
        x_angle_list.append(x0/d)
        y_angle_list.append(y0/d)
        tz.append(d)

        zstart = zi

    return x_angle_list, y_angle_list, tz

def interpolate_ray_paths_system(x_image, y_image, lens_system,
                                 include_substructure=True, realization=None, terminate_at_source=False,
                                 source_x=None, source_y=None):

    lens_model, kwargs_lens = lens_system.get_lensmodel(include_substructure, True, realization)
    zsource = lens_system.zsource

    return interpolate_ray_paths(x_image, y_image, lens_model, kwargs_lens, zsource,
                                 terminate_at_source, source_x, source_y)

def ddt_from_h(H, omega_matter, omega_matter_baryon, zlens, zsource):
    _astro = FlatLambdaCDM(H0=float(H), Om0=omega_matter, Ob0=omega_matter_baryon)
    lensCosmo = LensCosmo(zlens, zsource, _astro)
    Dd = lensCosmo.dd
    Ds = lensCosmo.ds
    Dds = lensCosmo.dds
    return (1 + zlens) * Dd * Ds / Dds

def interpolate_Ddt_h0(zlens, zsource, astropy_instance, h0_min=0.1, h0_max=300, steps=250):

    h0_values = np.linspace(h0_min, h0_max, steps)
    ddt = [ddt_from_h(hi, astropy_instance.Om0, astropy_instance.Ob0, zlens, zsource) for hi in h0_values]
    interpolator = interp1d(ddt, h0_values)
    return interpolator

def solve_H0_from_Ddt(zlens, zsource, D_dt, astropy_instance_ref, interpolation_function=None):

    omega_matter = astropy_instance_ref.Om0
    omega_matter_baryon = astropy_instance_ref.Ob0
    out = []

    def _func_to_min(h0, d):
        return (ddt_from_h(h0, omega_matter, omega_matter_baryon, zlens, zsource) - d) ** 2

    if isinstance(D_dt, list) or isinstance(D_dt, np.ndarray):

        for counter, di in enumerate(D_dt):
            if interpolation_function is None:
                result = minimize(_func_to_min, x0=73.3,
                              method='Nelder-Mead', args=di)['x'][0]

            else:
                try:
                    result = interpolation_function(di)
                except:

                    result = minimize(_func_to_min, x0=73.3,
                             method='Nelder-Mead', args=di)['x'][0]

            out.append(result)

    else:
        if interpolation_function is None:
            out = minimize(_func_to_min, x0=73.3,
                          method='Nelder-Mead', args=D_dt)['x'][0]
        else:
            try:
                out = interpolation_function(D_dt)
            except:
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

class AdaptiveGrid(object):

    def __init__(self, end_rmax, grid_resolution, theta, x_center, y_center):

        full_grid = RayShootingGrid(end_rmax, grid_resolution, theta)

        xgrid_0, ygrid_0 = full_grid.grid_at_xy_unshifted
        self.r_base = np.sqrt(xgrid_0 ** 2 + ygrid_0 ** 2).ravel()
        self.rmax = end_rmax
        self.grid_res = grid_resolution

        self.xgrid, self.ygrid = full_grid.grid_at_xy(x_center, y_center)

        self._pixels_per_axis = int(len(self.xgrid) ** 0.5)
        self.flux_values = np.zeros_like(self.xgrid)

    def get_indicies(self, rmin, rmax):

        condition = np.logical_and(self.r_base >= rmin, self.r_base < rmax)
        inds = np.where(condition)

        return inds

    def get_coordinates(self, rmin, rmax):

        indicies = self.get_indicies(rmin, rmax)

        return self.xgrid[indicies], self.ygrid[indicies], indicies

    def set_flux_in_pixels(self, pixel_indicies, flux_in_pixels):

        self.flux_values[pixel_indicies] = flux_in_pixels

    @property
    def image(self):
        return self.flux_values.reshape(self._pixels_per_axis, self._pixels_per_axis)





