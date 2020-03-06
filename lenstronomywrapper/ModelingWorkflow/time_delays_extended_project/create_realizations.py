from lenstronomywrapper.ModelingWorkflow.time_delays_extended_project.lens_analog_model import AnalogModel
import subprocess
import os
import sys

from MagniPy.Workflow.grism_lenses.he0435 import Lens0435
#from MagniPy.Workflow.grism_lenses.he0435_satellite_convention_phys import Lens0435 as
from MagniPy.Workflow.grism_lenses.lens2033 import WFI2033
from MagniPy.Workflow.grism_lenses.rxj1131 import Lens1131
from MagniPy.Workflow.radio_lenses.lens1115 import Lens1115
from MagniPy.Workflow.grism_lenses.lens2038 import Lens2038
from MagniPy.Workflow.grism_lenses.lens1608 import Lens1608
from MagniPy.Workflow.grism_lenses.DESJ0408 import Lens0408

def create_directory(dirname=''):

    proc = subprocess.Popen(['mkdir', dirname])
    proc.wait()

def create_realizations(lens_class, fname, log_mlow, window_size,
        subtract_exact_mass_sheets=False, Nreal=200):

    mdef = 'TNFW'
    realization_kwargs = {'mdef_main': mdef, 'mdef_los': mdef,
                          'log_mlow': log_mlow, 'log_mhigh': 10., 'power_law_index': -1.9,
                          'parent_m200': 10 ** 13, 'r_tidal': '0.5Rs',
                          'cone_opening_angle': 10*window_size, 'opening_angle_factor': 10*window_size,
                          'sigma_sub': 0.02, 'subtract_exact_mass_sheets': subtract_exact_mass_sheets,
                          'subtract_subhalo_mass_sheet': True, 'subhalo_mass_sheet_scale': 1,
                          'LOS_normalization': 1.}

    kwargs_cosmo = {'cosmo_kwargs': {'H0': 73.3}}
    lens_analog_model_class = AnalogModel(lens_class, kwargs_cosmo)

    save_name_path = os.getenv('HOME') + '/Code/tdelay_output/raw/' + fname + '/realizations/'

    if not os.path.exists(save_name_path):
        create_directory(save_name_path)

    for i in range(138, 140):
        realization_file_name = save_name_path + 'realization_'+str(i+1)+'.txt'
        realization = lens_analog_model_class.pyhalo.render('composite_powerlaw', realization_kwargs)[0]
        realization.save_to_file(realization_file_name, log_mlow)

lens_classes = [Lens1131(), Lens1115(), Lens0435(), Lens1608(), WFI2033(), Lens0408()]
lens_names = ['lens1131', 'lens1115', 'lens0435', 'lens1608', 'lens2033', 'lens0408']
sizes = [3., 1.8, 2., 1.9, 2.5, 3.8]

for i, (lens, fname, size) in enumerate(zip(lens_classes, lens_names, sizes)):
    if int(sys.argv[1]) == i+1:
        create_realizations(lens, fname, 6.7, size)


