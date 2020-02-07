from MagniPy.Workflow.grism_lenses.J0405 import J0405
from MagniPy.Workflow.grism_lenses.lens1606 import Lens1606
from MagniPy.Workflow.grism_lenses.he0435 import Lens0435
from MagniPy.Workflow.grism_lenses.lens2033 import WFI2033
from MagniPy.Workflow.grism_lenses.rxj1131 import Lens1131
from MagniPy.Workflow.radio_lenses.lens1115 import Lens1115
from MagniPy.Workflow.grism_lenses.lens2038 import Lens2038
from MagniPy.Workflow.grism_lenses.lens1608 import Lens1608
from MagniPy.Workflow.grism_lenses.DESJ0408 import Lens0408
from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.scripts import *
import os
from time import time
import sys

n_lens = int(sys.argv[1])

if n_lens < 1001:
    lens_name = 'lens1131'
    counter_start = 501
    counter_end = 1001
    N_start_0 = 0

elif n_lens < 2001:
    lens_name = 'lens1115'
    counter_start = 1501
    counter_end = 2001
    N_start_0 = 1000

elif n_lens < 3001:
    lens_name = 'lens0435'
    counter_start = 2501
    counter_end = 3001
    N_start_0 = 2000

elif n_lens < 4001:
    lens_name = 'lens1608'
    counter_start = 3501
    counter_end = 4001
    N_start_0 = 3000

elif n_lens < 5001:
    lens_name = 'lens2033'
    counter_start = 4501
    counter_end = 5001
    N_start_0 = 4000

elif n_lens < 6001:
    lens_name = 'lens0408'
    counter_start = 5501
    counter_end = 6001
    N_start_0 = 5000

fnames = ['lens1131', 'lens1115', 'lens0435', 'lens1608', 'lens2033', 'lens0408']
half_window_sizes = [3., 1.8, 2., 1.9, 2.5, 3.5]
lens_classes = [Lens1131(), Lens1115(), Lens0435(), Lens1608(), WFI2033(), Lens0408()]

for i, (name, _) in enumerate(zip(fnames, lens_classes)):
    if name == lens_name:
        break
else:
    raise Exception('lens name '+lens_name+' not found')

fname = fnames[i]
lens_class = lens_classes[i]
window_size = half_window_sizes[i]

print(fname, lens_classes[i], window_size)

mdef = 'TNFW'

N = 2
log_mlow = 7.
opening_angle = 8*window_size
time_delay_like = True
position_sigma = [0.005] * 4
gamma_prior_scale = 0.025
fix_D_dt = False

arrival_time_sigma = [delta_ti / ti for delta_ti, ti in
                      zip(lens_class.delta_time_delay, lens_class.relative_arrival_times)]
arrival_time_sigma = np.absolute(np.round(arrival_time_sigma, 5))

fit_smooth_kwargs = {'n_particles': 100, 'n_iterations': 200, 'n_run': 50, 'walkerRatio': 4, 'n_burn': 500}

if n_lens < counter_start:

    Nstart = n_lens - N_start_0

    print('SAMPLING WITH NO SUBSTRUCTURE...... ')
    SHMF_norm = 0.0
    LOS_norm = 0.
    t0 = time()
    save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/control_varyDdt/'
    if not os.path.exists(save_name_path):
        create_directory(save_name_path)
    run_real(lens_class, save_name_path, N, Nstart, SHMF_norm, LOS_norm, log_mlow, opening_angle,
             arrival_time_sigma,
             position_sigma, gamma_prior_scale, fix_D_dt, window_size, time_delay_like=True,
             fit_smooth_kwargs=fit_smooth_kwargs)
    tend = time()
    print('COMPLETED IN ' + str(np.round(tend - t0, 1)) + ' SECONDS')

else:

    Nstart = n_lens - N_start_0 - counter_start

    print('SAMPLING WITH SUBSTRUCTURE...... ')
    SHMF_norm = 0.75
    LOS_norm = 1.
    t0 = time()
    save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/LOSsubs_varyDdt/'
    if not os.path.exists(save_name_path):
        create_directory(save_name_path)
    run_real(lens_class, save_name_path, N, Nstart, SHMF_norm, LOS_norm, log_mlow, opening_angle,
             arrival_time_sigma,
             position_sigma, gamma_prior_scale, fix_D_dt, window_size, time_delay_like=True,
             fit_smooth_kwargs=fit_smooth_kwargs)
    tend = time()
    print('COMPLETED IN ' + str(np.round(tend - t0, 1)) + ' SECONDS')
