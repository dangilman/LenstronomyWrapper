from MagniPy.Workflow.grism_lenses.he0435 import Lens0435
from MagniPy.Workflow.grism_lenses.lens2033 import WFI2033
from MagniPy.Workflow.grism_lenses.rxj1131 import Lens1131
from MagniPy.Workflow.radio_lenses.lens1115 import Lens1115
from MagniPy.Workflow.grism_lenses.lens2038 import Lens2038
from MagniPy.Workflow.grism_lenses.lens1608 import Lens1608
from MagniPy.Workflow.grism_lenses.DESJ0408 import Lens0408
from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.run_script_general import run as run_lens
import os
from time import time
import sys

n_lens = int(sys.argv[1])
#n_lens = 1.
log_mlow = 6.7

if n_lens < 2001:
    Nstart = n_lens
    lens_name = 'lens1131'
    half_window_size = 3.
    lens_class = Lens1131()
    gamma_macro = 1.98

elif n_lens < 4001:
    Nstart = n_lens - 2000
    lens_name = 'lens1115'
    half_window_size = 1.8
    lens_class = Lens1115()
    gamma_macro = 2.2

elif n_lens < 6001:
    Nstart = n_lens - 4000
    lens_name = 'lens0435'
    half_window_size = 2.
    lens_class = Lens0435()
    gamma_macro = 1.93

elif n_lens < 8001:
    Nstart = n_lens - 6000
    lens_name = 'lens1608'
    half_window_size = 1.9
    lens_class = Lens1608()
    gamma_macro = 2.08

elif n_lens < 10001:
    Nstart = n_lens - 8000
    lens_name = 'lens2033'
    half_window_size = 2.5
    lens_class = WFI2033()
    gamma_macro = 1.95

elif n_lens < 12001:
    Nstart = n_lens - 10000
    lens_name = 'lens0408'
    half_window_size = 3.5
    lens_class = Lens0408()
    gamma_macro = 1.9

else:
    raise Exception('out of range.')

run_lens(Nstart, lens_class, lens_name, log_mlow, half_window_size, gamma_macro)
