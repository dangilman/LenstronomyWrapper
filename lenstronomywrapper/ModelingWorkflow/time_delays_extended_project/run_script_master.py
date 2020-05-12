from MagniPy.Workflow.grism_lenses.he0435 import Lens0435
#from MagniPy.Workflow.grism_lenses.he0435_satellite_convention_phys import Lens0435 as
from MagniPy.Workflow.grism_lenses.lens2033 import WFI2033
from MagniPy.Workflow.grism_lenses.rxj1131 import Lens1131
from MagniPy.Workflow.radio_lenses.lens1115 import Lens1115
from MagniPy.Workflow.grism_lenses.lens2038 import Lens2038
from MagniPy.Workflow.grism_lenses.lens1608 import Lens1608
from MagniPy.Workflow.grism_lenses.DESJ0408 import Lens0408
from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.run_script_general import run as run_lens
import os
import numpy as np
import time
import sys

#n_lens = 1.

#time.sleep(180)

def index_read(idx):
    if idx < 51:
        return False
    elif idx < 101:
        return True
    elif idx < 301:
        return False
    else:
        return True

do_sampling_with_no_shapelets = False

class GammaPrior(object):

    def __init__(self, mean, sigma):

        self.mean, self.sigma = mean, sigma

    def __call__(self):

        return np.random.uniform(self.mean - 2*self.sigma, self.mean + 2*self.sigma)

def lens1131_exposure(index):
    vary_shapelets = index_read(index)

    if vary_shapelets:
        # DONE
        return 2000, 0.37, True
    else:
        # run again
        return 2000, 0.45, do_sampling_with_no_shapelets

def lens1115_exposure(index):
    vary_shapelets = index_read(index)

    if vary_shapelets:
        # DONE
        return 4000, 0.43, True
    else:
        # DONE
        return 4000, 0.575, do_sampling_with_no_shapelets

def lens0435_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        # DONE
        return 4000, 0.07, True
    else:
        # DONE
        return 4000, 0.13, do_sampling_with_no_shapelets

def lens1608_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        # run again
        return 20000, 0.34, True
    else:
        # DONE
        return 20000, 0.5, do_sampling_with_no_shapelets

def lens2033_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        # DONE
        return 3000, 0.17, True
    else:
        # DONE
        return 3000, 0.25, do_sampling_with_no_shapelets

def lens0408_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        # DONE
        return 3000, 0.15, True
    else:
        # DONE
        return 3000, 0.35, do_sampling_with_no_shapelets

n_lens_start = int(sys.argv[1])

try:
    n_lens_end = int(sys.argv[2])
    assert n_lens_end is not None
    assert n_lens_end > n_lens_start
except:
    n_lens_end = n_lens_start + 1

for n_lens in range(n_lens_start, n_lens_end):

    if n_lens < 601:
        Nstart = n_lens
        lens_name = 'lens1131'
        half_window_size = 3.
        lens_class = Lens1131()
        # DONE
        gamma_mean, gamma_sigma = 1.98, 0.026
        exp_time, background_rms, do_sampling = lens1131_exposure(Nstart)

    elif n_lens < 1001:
        Nstart = n_lens - 500
        lens_name = 'lens1115'
        half_window_size = 1.8
        lens_class = Lens1115()
        gamma_mean, gamma_sigma = 2.2, 0.07
        exp_time, background_rms, do_sampling = lens1115_exposure(Nstart)

    elif n_lens < 1501:
        Nstart = n_lens - 1000
        lens_name = 'lens0435'
        half_window_size = 2.
        lens_class = Lens0435()
        gamma_mean, gamma_sigma = 1.93, 0.024
        exp_time, background_rms, do_sampling = lens0435_exposure(Nstart)

    elif n_lens < 2001:
        Nstart = n_lens - 1500
        lens_name = 'lens1608'
        half_window_size = 1.9
        lens_class = Lens1608()
        gamma_mean, gamma_sigma = 2.08, 0.025
        exp_time, background_rms, do_sampling = lens1608_exposure(Nstart)

    elif n_lens < 2501:
        Nstart = n_lens - 2000
        lens_name = 'lens2033'
        half_window_size = 2.5
        lens_class = WFI2033()
        gamma_mean, gamma_sigma = 1.95, 0.032
        exp_time, background_rms, do_sampling = lens2033_exposure(Nstart)

    elif n_lens < 3001:
        Nstart = n_lens - 2500
        lens_name = 'lens0408'
        half_window_size = 3.5
        lens_class = Lens0408()
        gamma_mean, gamma_sigma = 1.9189, 0.0035
        exp_time, background_rms, do_sampling = lens0408_exposure(Nstart)

    else:
        raise Exception('out of range.')

    fit_smooth_kwargs = {'n_particles': 150, 'n_iterations': 350, 'n_run': 150,
                         'walkerRatio': 4, 'n_burn': 600}
    fit_smooth_kwargs = {'n_particles': 2, 'n_iterations': 2, 'n_run': 4, 'walkerRatio': 4,
                         'n_burn': 10}
    log_mlow = 8.

    name_append = '_mlow6.7'

    sample_gamma = True

    if sample_gamma:
        name_append += '_samplegammawide'
        gamma_sigma *= 3
    else:
        gamma_sigma = 1e-9

    gamma_prior = GammaPrior(gamma_mean, gamma_sigma)
    # window_scale = 10
    window_scale = 10
    run_lens(Nstart, lens_class, gamma_prior, lens_name, log_mlow, half_window_size, exp_time,
             background_rms=background_rms, subtract_exact_mass_sheets=False, name_append=name_append,
             fix_Ddt=True, fit_smooth_kwargs=fit_smooth_kwargs, window_scale=window_scale,
             do_sampling=do_sampling)
