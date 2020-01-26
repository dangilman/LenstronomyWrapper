from lenstronomy.Util.param_util import shear_cartesian2polar, ellipticity2phi_q
import numpy as np

def kwargs_to_array(kwargs):

    array = [kwargs[key] for key in kwargs.keys()]
    return np.array(array)

def kwargs_e1e2_to_polar(kwargs):

    out = {}
    phi, q = ellipticity2phi_q(kwargs['e1'], kwargs['e2'])
    ellip_PA = phi * 180/np.pi
    ellip = 1-q

    for key in kwargs:
        if key == 'e1':
            out['ellip'] = ellip
        elif key == 'e2':
            out['ellip_PA'] = ellip_PA
        else:
            out[key] = kwargs[key]

    return out

def kwargs_gamma1gamma2_to_polar(kwargs):

    out = {}
    phi, shear = shear_cartesian2polar(kwargs['gamma1'], kwargs['gamma2'])
    shear_PA = phi * 180 / np.pi

    for key in kwargs:
        if key == 'gamma1':
            out['shear'] = shear
        elif key == 'gamma2':
            out['shear_PA'] = shear_PA
        else:
            out[key] = kwargs[key]

    return out
