from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.lens_analog_model import AnalogModel
import os
import sys
import numpy as np
import subprocess

def create_directory(dirname=''):

    proc = subprocess.Popen(['mkdir', dirname])
    proc.wait()

def run_real(lens_analog_model_class, gamma_prior, save_name_path, N, N_start, gen_realization, arrival_time_sigma,
            image_positions_sigma, realization_kwargs,
             fix_D_dt, window_size, exp_time, background_rms, time_delay_like=True, fit_smooth_kwargs=None,
             shapelet_nmax=None):

    out = lens_analog_model_class.run(save_name_path, N_start, N, gen_realization, arrival_time_sigma,
                    image_positions_sigma, realization_kwargs, time_delay_like, fix_D_dt, fit_smooth_kwargs,
                    window_size, exp_time, background_rms, shapelet_nmax, gamma_prior)

