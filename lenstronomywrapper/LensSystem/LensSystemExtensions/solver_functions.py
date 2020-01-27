import numpy as np

def _rayshoot_pixels(xcoords, ycoords, lens_model,
                    kwargs_lens_model, light_model):

    light = light_model.surface_brightness(xcoords, ycoords, lens_model, kwargs_lens_model)
    #light = light_model.surface_brightness(light_model.sourceLight, light_model.kwargs_light, bx, by, kwargs_light_model)
    return light

def _get_new_estimate(image, xi, yi, xgrid, ygrid):
    max_image = np.where(image == np.amax(image))
    return xi + xgrid[max_image], yi + ygrid[max_image]

def _iterate_once(window, resolution, xi, yi, lensModel, kwargs_lens, light_model):
    npix = int(2 * window / resolution)
    x = y = np.linspace(-window, window, npix)
    xgrid, ygrid = np.meshgrid(x, y)
    xgrid, ygrid = xgrid.ravel(), ygrid.ravel()
    xcoords, ycoords = xi + xgrid, yi + ygrid
    image = _rayshoot_pixels(xcoords, ycoords, lensModel, kwargs_lens, light_model)
    xi, yi = _get_new_estimate(image, xi, yi, xgrid, ygrid)
    return xi[0], yi[0]

def _solve_one_image(windows, resolutions, xi, yi, lensModel, kwargs_lens, light_model):
    for w, res in zip(windows, resolutions):
        xi, yi = _iterate_once(w, res, xi, yi, lensModel, kwargs_lens, light_model)
    return xi, yi

def _source_plane_delta(vec, mod, kw, source_x, source_y):
    bx, by = mod.ray_shooting(vec[0:4], vec[4:], kw)
    dx = bx - source_x
    dy = by - source_y
    return np.sum(dx ** 2 + dy ** 2)

