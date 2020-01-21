from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.simulated_lens_model import SimulatedModel
import os
import sys
import subprocess

def create_directory(dirname=''):

    proc = subprocess.Popen(['mkdir', dirname])
    proc.wait()

def run(output_path, Nstart, N, SHMF_norm, LOS_norm, log_mlow, opening_angle, arrival_time_sigma,
        fix_D_dt, time_delay_like=True):

    mdef = 'TNFW'
    realization_kwargs = {'mdef_main': mdef, 'mdef_los': mdef,
                                  'log_mlow': log_mlow, 'log_mhigh': 10., 'power_law_index': -1.9,
                                  'parent_m200': 10**13, 'r_tidal': '0.5Rs',
                                  'cone_opening_angle': opening_angle, 'opening_angle_factor': 10,
                                  'sigma_sub': SHMF_norm,
                                  'subtract_subhalo_mass_sheet': True, 'subhalo_mass_sheet_scale': 1,
                                  'LOS_normalization': LOS_norm}

    sim = SimulatedModel('composite_powerlaw', realization_kwargs)
    sim.run_and_save(output_path, Nstart, N, arrival_time_sigma, time_delay_like, fix_D_dt)

# output_path = os.getenv('HOME') + '/data/mock_data/simulated_time_delays/extended_sim_control/'
# if not os.path.exists(output_path):
#     create_directory(output_path)
# arrival_time_sigma = [0.01] * 3
# fix_d_dt = True
# SHMF_norm = 0.
# LOS_norm = 0.
# log_mlow = 7.
# opening_angle = 12.
# N = 20
# Nstart = 1 + N * (int(sys.argv[1]) - 1)
# print(Nstart)
# run(output_path, Nstart, N, SHMF_norm, LOS_norm, log_mlow, opening_angle, arrival_time_sigma, fix_d_dt)
