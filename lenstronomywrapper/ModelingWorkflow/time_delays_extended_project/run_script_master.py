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
import time
import sys

n_lens = int(sys.argv[1])
#n_lens = 1.
log_mlow = 6.7
#time.sleep(180)

if n_lens < 501:
    Nstart = n_lens
    lens_name = 'lens1131'
    half_window_size = 3.
    lens_class = Lens1131()
    exp_time = 1000.
    background_rms = 0.3

elif n_lens < 1001:
    Nstart = n_lens - 500
    lens_name = 'lens1115'
    half_window_size = 1.8
    lens_class = Lens1115()
    exp_time = 2000.
    background_rms = 0.3

elif n_lens < 1501:
    Nstart = n_lens - 1000
    lens_name = 'lens0435'
    half_window_size = 2.
    lens_class = Lens0435()
    exp_time = 1800
    background_rms = 0.3

elif n_lens < 2001:
    Nstart = n_lens - 1500
    lens_name = 'lens1608'
    half_window_size = 1.9
    lens_class = Lens1608()
    exp_time = 2050
    background_rms = 0.3

elif n_lens < 2501:
    Nstart = n_lens - 2000
    lens_name = 'lens2033'
    half_window_size = 2.5
    lens_class = WFI2033()
    exp_time = 2050
    background_rms = 0.3

elif n_lens < 3001:
    Nstart = n_lens - 2500
    lens_name = 'lens0408'
    half_window_size = 3.8
    lens_class = Lens0408()
    exp_time = 2000
    background_rms = 0.3

else:
    raise Exception('out of range.')

run_lens(Nstart, lens_class, lens_name, log_mlow, half_window_size, exp_time,
         background_rms=background_rms, subtract_exact_mass_sheets=False, name_append='')
