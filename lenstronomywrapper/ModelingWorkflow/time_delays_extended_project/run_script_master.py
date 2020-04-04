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


def index_read(idx):
    if idx < 51:
        return False
    elif idx < 101:
        return True
    elif idx < 301:
        return False
    else:
        return True

def lens1131_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 1500, 0.7
    else:
        return 1500, 0.8

def lens1115_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 4000, 0.65
    else:
        return 4000, 0.75

def lens0435_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 50000, 0.12
    else:
        return 30000, 0.16

def lens1608_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 20000, 0.3
    else:
        return 20000, 0.4

def lens2033_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 3000, 0.5
    else:
        return 3000, 0.55

def lens0408_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 60000, 0.08
    else:
        return 60000, 0.1

if n_lens < 501:
    Nstart = n_lens
    lens_name = 'lens1131'
    half_window_size = 3.
    lens_class = Lens1131()
    # DONE
    exp_time, background_rms = lens1131_exposure(Nstart)

elif n_lens < 1001:
    Nstart = n_lens - 500
    lens_name = 'lens1115'
    half_window_size = 1.8
    lens_class = Lens1115()
    exp_time, background_rms = lens1115_exposure(Nstart)

elif n_lens < 1501:
    Nstart = n_lens - 1000
    lens_name = 'lens0435'
    half_window_size = 2.
    lens_class = Lens0435()
    exp_time, background_rms = lens0435_exposure(Nstart)

elif n_lens < 2001:
    Nstart = n_lens - 1500
    lens_name = 'lens1608'
    half_window_size = 1.9
    lens_class = Lens1608()
    exp_time, background_rms = lens1608_exposure(Nstart)

elif n_lens < 2501:
    Nstart = n_lens - 2000
    lens_name = 'lens2033'
    # DONE
    half_window_size = 2.5
    lens_class = WFI2033()
    exp_time, background_rms = lens2033_exposure(Nstart)

elif n_lens < 3001:
    Nstart = n_lens - 2500
    lens_name = 'lens0408'
    half_window_size = 3.8
    lens_class = Lens0408()
    exp_time, background_rms = lens0408_exposure(Nstart)

else:
    raise Exception('out of range.')

run_lens(Nstart, lens_class, lens_name, log_mlow, half_window_size, exp_time,
         background_rms=background_rms, subtract_exact_mass_sheets=False, name_append='',
         fix_Ddt=True)
