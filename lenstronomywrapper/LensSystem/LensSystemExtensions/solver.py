import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from scipy.optimize import minimize
from lenstronomywrapper.LensSystem.LensSystemExtensions.solver_functions import _solve_one_image, _source_plane_delta

from lenstronomy.LightModel.light_model import LightModel

def iterative_rayshooting(source_x, source_y, x_guess, y_guess, lensModel, kwargs_lens,
                          window_sizes=None, grid_resolutions=None):

    light_model = LightModel(['GAUSSIAN'])
    kwargs_light = [{'center_x': source_x, 'center_y': source_y, 'sigma': 0.002, 'amp':1}]

    if window_sizes is None:
        window_sizes = [0.15, 0.07, 0.03]
    if grid_resolutions is None:
        grid_resolutions = [0.005, 0.004, 0.0005]

    if len(window_sizes) != len(grid_resolutions):
        raise Exception('Length of window size and grid resolution lists must be equal')

    if len(x_guess) != len(y_guess):
        raise Exception('Length of image arrays must match.')

    x_image_out, y_image_out = [], []
    for (xi, yi) in zip(x_guess, y_guess):
        xout, yout = _solve_one_image(window_sizes, grid_resolutions, xi, yi, lensModel, kwargs_lens, light_model, kwargs_light)
        x_image_out.append(xout)
        y_image_out.append(yout)

    return np.array(x_image_out), np.array(y_image_out)

def iterative_rayshooting_simplex(source_x, source_y, x_guess, y_guess, lensModel, kwargs_lens,
                                  window_sizes=None, grid_resolutions=None):

    x_guess, y_guess = iterative_rayshooting(source_x, source_y, x_guess, y_guess, lensModel, kwargs_lens, window_sizes, grid_resolutions)

    x_image, y_image = simplex(source_x, source_y, x_guess, y_guess, lensModel, kwargs_lens)

    return x_image, y_image

def simplex(source_x, source_y, x_guess, y_guess, lensModel, kwargs_lens, precision_limit=10**-5):

    img_vec = np.append(x_guess, y_guess)

    result = minimize(_source_plane_delta, x0=img_vec, args=(lensModel, kwargs_lens, source_x, source_y),
                      tol=precision_limit**2,
                      method='Nelder-Mead')['x']

    x_image, y_image = result[0:4], result[4:]

    return x_image, y_image

def iterative_simplex(lens_system, source_x, source_y, halo_masses,
                      precision_limit=10 ** -4, verbose=False):

    lens_system.update_source_centroid(source_x, source_y)
    lensModel, kwargs = lens_system.get_lensmodel(include_substructure=False)
    x_image, y_image = lens_system.solve_lens_equation(lensModel, kwargs, precision_limit)

    realization = lens_system.realization
    n_halos_last = -1

    for cut in halo_masses:
        if verbose:
            print(x_image)
            print(y_image)
            print('solving with halos larger than 10^' + str(np.log10(cut)) + '... ')
        real = realization.filter_by_mass(cut)
        halo_names, halo_redshifts, kwargs_halos, kwargs_lenstronomy = real.lensing_quantities()
        halo_redshifts = list(halo_redshifts)
        lens_model_names, macro_redshifts, macro_kwargs, convention_index = lens_system.macromodel.get_lenstronomy_args()
        names = lens_model_names + halo_names
        redshifts = macro_redshifts + halo_redshifts

        n_halos = len(halo_names)
        if n_halos == n_halos_last:
            break

        if verbose: print('n halos:', n_halos)
        kwargs = macro_kwargs + kwargs_halos
        lensModel = LensModel(names, lens_redshift_list=redshifts, z_lens=lens_system.zlens, z_source=lens_system.zsource,
                              multi_plane=True, numerical_alpha_class=kwargs_lenstronomy,
                              observed_convention_index=convention_index, cosmo=lens_system.astropy)
        x_image, y_image = simplex(source_x, source_y, x_image, y_image, lensModel, kwargs, precision_limit)
        n_halos_last = n_halos
    if verbose:
        print(x_image)
        print(y_image)
    return x_image, y_image
