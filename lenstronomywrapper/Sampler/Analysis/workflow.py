from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensSystem.BackgroundSource.quasar import Quasar
from lenstronomywrapper.LensSystem.BackgroundSource.double_gaussian import DoubleGaussian
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.LensSystem.LensComponents.SIS import SISsatellite
from lenstronomywrapper.LensSystem.LensSystemExtensions.lens_maps import LensMaps, ResidualLensMaps

from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
from MagniPy.Workflow.grism_lenses.lens2038 import Lens2038 as Lens
from MagniPy.Workflow.grism_lenses.b1422 import Lens1422 as Lens
#from MagniPy.Workflow.grism_lenses.he0435 import Lens0435 as Lens
from lenstronomywrapper.LensSystem.LensComponents.multipole import Multipole

from pyHalo.pyhalo import pyHalo
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

mass_definition = 'TNFW' # black holes are by definition point masses
lens_cone_opening_angle = 4 # in arcsec; typically Einstein radius is ~1 arcsec so this renders out to radius 3*R_ein
lens = Lens()
zlens, zsource = lens.zlens, lens.zsrc
print(lens.zlens, lens.zsrc)
# set the lens and source redshifts. Typical configurations are zlens ~0.5 and zsource ~1.5-2.0

# Set the redshift range in which to render point masses
kwargs_halo_mass_function = {'geometry_type': 'DOUBLE_CONE'}
pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_halo_mass_function)

log_mlow = 6.
log_mhigh = 10
power_law_index = -1.9
delta_power_law_index = -0.
LOS_norm = 0.8
m_pivot = 10**8
kwargs_halo_mass_function1 = {'geometry_type': 'DOUBLE_CONE'}
pyhalo = pyHalo(zlens, zsource, kwargs_halo_mass_function=kwargs_halo_mass_function1)

SHMF_norm = 0.025
LOS_norm = 0.8

realization_kwargs_1 = {'mass_func_type': 'POWER_LAW', 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                      'log_mass_sheet_min': 7.5, 'log_mass_sheet_max': log_mhigh, 'mdef_main': 'TNFW','mdef_los': 'TNFW', 'sigma_sub': SHMF_norm,
                      'delta_power_law_index': delta_power_law_index, 'cone_opening_angle': lens_cone_opening_angle,
                      'log_m_host': 13., 'power_law_index': -1.9, 'm_pivot': m_pivot, 'r_tidal': 10,
                      'LOS_normalization': LOS_norm, 'subhalo_convergence_correction_profile':'NFW'}
realization_kwargs_2 = {'mass_func_type': 'POWER_LAW', 'log_mlow': log_mlow, 'log_mhigh': log_mhigh,
                      'log_mass_sheet_min': log_mlow, 'log_mass_sheet_max': log_mhigh, 'mdef_main': 'TNFW','mdef_los': 'TNFW', 'sigma_sub': SHMF_norm,
                      'delta_power_law_index': delta_power_law_index, 'cone_opening_angle': lens_cone_opening_angle,
                      'log_m_host': 13., 'power_law_index': -1.9, 'm_pivot': m_pivot, 'r_tidal': 10,
                      'LOS_normalization': LOS_norm, 'subhalo_convergence_correction_profile':'NFW'}

realization_type = 'composite_powerlaw'

def solve_single(realizations):

    (realization_1, realization_2) = realizations

    kwargs_lens_1 = [{'theta_E': 1., 'center_x': 0., 'center_y': -0.0, 'e1': 0.1, 'e2': 0.2, 'gamma': 2.0},
                   {'gamma1': 0.0, 'gamma2': 0.03}]
    kwargs_lens_2 = [{'theta_E': 1., 'center_x': 0., 'center_y': -0.0, 'e1': 0.1, 'e2': 0.2, 'gamma': 2.0},
                     {'gamma1': 0.0, 'gamma2': 0.03}]

    source_size_parsec = 60.  # FWHM of a Gaussian
    kwargs_source = {'center_x': 0., 'center_y': 0., 'source_fwhm_pc': source_size_parsec}

    main_lens_fit = PowerLawShear(zlens, kwargs_lens_1)

    macromodel_fit = MacroLensModel([main_lens_fit])
    background_quasar = Quasar(kwargs_source)
    data_to_fit = lens
    lens_system_quad_1 = QuadLensSystem.shift_background_auto(data_to_fit, macromodel_fit,
                                                              zsource, background_quasar, realization_1,
                                                              None)

    # optimizer = HierarchicalOptimization(lens_system_quad_1, settings_class='default_CDM')
    # kwargs_lens_final_approx, lens_model_full_approx, _ = optimizer.optimize(data_to_fit,
    #                                                                          param_class_name='free_shear_powerlaw',
    #                                                                          constrain_params=None, verbose=verbose)
    optimizer_brute = BruteOptimization(lens_system_quad_1)
    kwargs_lens_final_approx, lens_model_full_approx, _ = optimizer_brute.optimize(data_to_fit, 'free_shear_powerlaw',
                                                                     None, verbose, True, {})

    main_lens = PowerLawShear(zlens, kwargs_lens_2)
    macromodel_fit = MacroLensModel([main_lens])
    background_quasar = Quasar(kwargs_source)
    lens_system_quad_2 = QuadLensSystem.shift_background_auto(data_to_fit, macromodel_fit,
                                                              zsource, background_quasar, realization_2,
                                                              None)

    optimizer_brute = BruteOptimization(lens_system_quad_2)
    #optimizer = HierarchicalOptimization(lens_system_quad_1, settings_class='default')
    kwargs_lens_final, lens_model_full, _ = optimizer_brute.optimize(data_to_fit, 'free_shear_powerlaw',
                                                        None, verbose, True, {})

    lensmodel_approx, kwargs_approx = lens_system_quad_1.get_lensmodel()
    magnifications_approx, _ = lens_system_quad_1.quasar_magnification(lens.x,
                                                                       lens.y, lens_model=lensmodel_approx,
                                                                       kwargs_lensmodel=kwargs_approx, normed=True,
                                                                       adaptive=True, verbose=verbose)


    lensmodel, kwargs = lens_system_quad_2.get_lensmodel()
    magnifications, _ = lens_system_quad_2.quasar_magnification(lens.x,
                                                                lens.y, lens_model=lensmodel,
                                                                kwargs_lensmodel=kwargs, normed=True, adaptive=True,
                                                                verbose=verbose)

    ratios1 = magnifications_approx/magnifications_approx[0]
    ratios2 = magnifications/magnifications[0]
    return ratios1[1:], ratios2[1:]


niter = 8
flux_out_1 = None
flux_out_2 = None
verbose = False

output_filename = 'b1422_ratios'

for j in range(0, niter):

    realization_list = []

    r1 = pyhalo.render(realization_type, realization_kwargs_1, nrealizations=1)[0]
    r2 = pyhalo.render(realization_type, realization_kwargs_2, nrealizations=1)[0]

    f1, f2 = solve_single([r1, r2])

    if flux_out_1 is None:
        flux_out_1 = f1
        flux_out_2 = f2
    else:
        flux_out_1 = np.vstack((flux_out_1, f1))
        flux_out_2 = np.vstack((flux_out_2, f2))

with open(output_filename + '_noapprox.txt', 'a') as f:
    for i in range(0, niter):
        for j in range(0, flux_out_1.shape[1]):
            f.write(str(flux_out_1[i, j])+" ")
        f.write('\n')
with open(output_filename + '_noapprox.txt', 'a') as f:
    for i in range(0, niter):
        for j in range(0, flux_out_2.shape[1]):
            f.write(str(flux_out_2[i, j])+" ")
        f.write('\n')



