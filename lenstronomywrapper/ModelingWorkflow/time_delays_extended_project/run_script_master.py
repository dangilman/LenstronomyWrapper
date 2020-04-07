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

#n_lens = 1.
log_mlow = 8.7
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
        # DONE
        return 2000, 0.5
    else:
        # run again
        return 2000, 0.65

def lens1115_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 4000, 0.45
    else:
        return 4000, 0.5

def lens0435_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        # run again
        return 50000, 0.1
    else:
        # DONE
        return 30000, 0.13

def lens1608_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        # run again
        return 20000, 0.43
    else:
        # run again
        return 20000, 0.53

def lens2033_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        return 3000, 0.5
    else:
        return 3000, 0.5

def lens0408_exposure(index):
    vary_shapelets = index_read(index)
    if vary_shapelets:
        # DONE
        return 60000, 0.08
    else:
        # DONE
        return 60000, 0.1

run_control = True
run_control_shapelets = True
run_los_plus_subs = True
run_los_plus_subs_shapelets = True

n_lens_start = int(sys.argv[1])

try:
    n_lens_end = int(sys.argv[2])
    assert n_lens_end is not None
    assert n_lens_end > n_lens_start
except:
    n_lens_end = n_lens_start + 1

for n_lens in range(n_lens_start, n_lens_end):

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

    if Nstart > 0 and Nstart < 51:
        if not run_control: exit(1)
    if Nstart > 50 and Nstart < 101:
        if not run_control_shapelets: exit(1)
    if Nstart > 100 and Nstart < 301:
        if not run_los_plus_subs: exit(1)
    if Nstart > 300 and Nstart < 501:
        if not run_los_plus_subs_shapelets: exit(1)

    run_lens(Nstart, lens_class, lens_name, log_mlow, half_window_size, exp_time,
             background_rms=background_rms, subtract_exact_mass_sheets=False, name_append='',
             fix_Ddt=True)
