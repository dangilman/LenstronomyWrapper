from lenstronomy.LensModel.lens_model import LensModel
import numpy as np
from lenstronomywrapper.LensSystem.lens_base import LensBase
from lenstronomywrapper.Optimization.quad_optimization.brute import BruteOptimization
from lenstronomywrapper.Optimization.extended_optimization.source_reconstruction import SourceReconstruction
from pyHalo.Cosmology.cosmology import Cosmology

class ArcQuadLensSystem(LensBase):

    def __init__(self, macromodel, z_source, lens_light_model, source_light_model,
                 substructure_realization=None, pyhalo_cosmology=None):

        self.lens_light_model = lens_light_model
        self.source_light_model = source_light_model

        if pyhalo_cosmology is None:
            # the default cosmology in pyHalo, currently WMAP9
            pyhalo_cosmology = Cosmology()

        self.pc_per_arcsec_zsource = 1000 / pyhalo_cosmology.astropy.arcsec_per_kpc_proper(z_source).value

        super(ArcQuadLensSystem, self).__init__(macromodel, z_source, substructure_realization, pyhalo_cosmology)

    def get_smooth_lens_system(self):

        arclens_smooth_component = ArcQuadLensSystem(self.macromodel, self.zsource,
                                                     self.lens_light_model, self.source_light_model,
                                                     None, self.pyhalo_cosmology)

        source_x, source_y = self.source_centroid_x, self.source_centroid_y
        light_x, light_y = self.light_centroid_x, self.light_centroid_y

        arclens_smooth_component.update_source_centroid(source_x, source_y)
        arclens_smooth_component.update_light_centroid(light_x, light_y)

        arclens_smooth_component._set_concentric()

        return arclens_smooth_component

    def get_lens_model_components(self):

        return self.macromodel, self.source_light_model, self.lens_light_model, self.background_quasar

    def add_source_light_component(self, new_component):

        self.source_light_model.add_component(new_component)

    def add_lens_light_component(self, new_component):

        self.lens_light_model.add_component(new_component)

    def add_macromodel_component(self, new_component):

        self.macromodel.add_component(new_component)

    @classmethod
    def fromQuad(cls, quad_lens_system, lens_light_model, source_light_model, inherit_substructure_realization=True):

        if inherit_substructure_realization:
            pass_realization = quad_lens_system.realization
        else:
            pass_realization = None

        arcquadlens = ArcQuadLensSystem(quad_lens_system.macromodel, quad_lens_system.zsource, lens_light_model,
                                        source_light_model, pass_realization, quad_lens_system.pyhalo_cosmology)

        source_x, source_y = quad_lens_system.source_centroid_x, quad_lens_system.source_centroid_y
        light_x, light_y = quad_lens_system.light_centroid_x, quad_lens_system.light_centroid_y

        arcquadlens.update_source_centroid(source_x, source_y)
        arcquadlens.update_light_centroid(light_x, light_y)

        arcquadlens._set_concentric()

        arcquadlens.position_convention_halo = quad_lens_system.position_convention_halo

        return arcquadlens

    def _set_concentric(self):

        for component in self.lens_light_model.components:

            if component.concentric_with_model is not None:
                idx = component.concentric_with_model
                model = self.macromodel.components[idx]

                for i in range(0, len(component._kwargs)):
                    component._kwargs[i]['center_x'], component._kwargs[i]['center_y'] = model.x_center, model.y_center

        for component_idx, component in enumerate(self.source_light_model.components):

            if component.concentric_with_source is not None:
                for i in range(0, len(component._kwargs)):

                    match_idx = component.concentric_with_source

                    if match_idx == 0:
                        component._kwargs[i]['center_x'] = self.source_centroid_x
                        component._kwargs[i]['center_y'] = self.source_centroid_y
                    else:
                        print(match_idx)
                        print(self.source_light_model.components)
                        comp = self.source_light_model.components[match_idx]

                        component._kwargs[i]['center_x'] = comp._source_x
                        component._kwargs[i]['center_y'] = comp._source_y

    def fit(self, data_to_fit, fit_sequence, **kwargs):

        optimizer = SourceReconstruction(self, data_to_fit, **kwargs)
        chain_list, kwargs_result, kwargs_model, multi_band_list, kwargs_special, param_class = optimizer.\
            optimize(fit_sequence)

        return chain_list, kwargs_result, kwargs_model, multi_band_list, kwargs_special, param_class

    def initialize(self, data_to_fit, opt_routine='fixed_powerlaw_shear', constrain_params=None, verbose=False):

        optimizer = BruteOptimization(self)

        _, _, _ = optimizer.optimize(data_to_fit, opt_routine, constrain_params, verbose, include_substructure=False)

        return

    def update_source_centroid(self, source_x, source_y):

        self.source_centroid_x = source_x
        self.source_centroid_y = source_y

    def update_light_centroid(self, light_x, light_y):

        self.light_centroid_x = light_x
        self.light_centroid_y = light_y

    def update_lens_light(self, new_lens_light_kwargs):

        count = 0
        for component in self.lens_light_model.components:
            ind1 = count
            ind2 = count + component.n_models
            kwargs_component = new_lens_light_kwargs[ind1:ind2]
            component._kwargs = kwargs_component
            count += component.n_models

    def update_source_light(self, new_source_light_kwargs):

        count = 0
        for component in self.source_light_model.components:
            ind1 = count
            ind2 = count + component.n_models
            kwargs_component = new_source_light_kwargs[ind1:ind2]
            component._kwargs = kwargs_component
            count += component.n_models

    def get_lens_light(self):

        instance, kwargs = self.lens_light_model.lensLight, self.lens_light_model.kwargs_light
        return instance, kwargs

    def get_source_light(self):

        instance, kwargs = self.source_light_model.sourceLight, self.source_light_model.kwargs_light
        return instance, kwargs

    def quasar_magnification(self, x, y, background_source,
                             lens_model,
                             kwargs_lensmodel, normed=True,
                             retry_if_blended=0,
                             enforce_unblended=False,
                             adaptive=False, verbose=False, point_source=False,
                             grid_axis_ratio=1):

        """
        Computes the magnifications (or flux ratios if normed=True)

        :param x: x image position
        :param y: y image position
        :param background_quasar: an instance of the background source light profile
        :param lens_model: an instance of LensModel (see lenstronomy.lens_model)
        :param kwargs_lensmodel: key word arguments for the lens_model
        :param normed: if True returns flux ratios
        :param retry_if_blended: a integer that specifies how many times to try
        increasing the size of the ray tracing window if an image comes out blended together
        :param point_source: if True, computes the magnification of a point source
        """

        if point_source:
            mags = lens_model.magnification(x, y, kwargs_lensmodel)
            mags = abs(mags)
            if normed:
                mags *= max(mags) ** -1
            return mags, False

        background_source.setup(self.pc_per_arcsec_zsource)
        if not hasattr(self, 'source_centroid_x') or self.source_centroid_x is None:
            raise Exception('lens system must have a specified source coordinate in order to compute the magnification'
                            'from an extended source.')
        background_source.update_position(self.source_centroid_x, self.source_centroid_y)
        relative_angles = [np.arctan2(-xi, yi) for (xi, yi) in zip(x, y)]

        return background_source.magnification(x, y, lens_model,
                                               kwargs_lensmodel,
                                               normed, retry_if_blended,
                                               enforce_unblended, adaptive, verbose,
                                               grid_axis_ratio,
                                               relative_angles)

    def plot_images(self, x, y, lens_model=None, kwargs_lensmodel=None):

        if lens_model is None or kwargs_lensmodel is None:
            lens_model, kwargs_lensmodel = self.get_lensmodel()

        return self.background_quasar.plot_images(x, y, lens_model, kwargs_lensmodel)

    def add_mcmc_model(self, kwargs_list, kwargs_model, kwargs_result,
                       multi_band_list, kwargs_special):

        self.mcmc_output_kwargs = {'kwargs_list': kwargs_list, 'kwargs_result': kwargs_result,
                                 'kwargs_model': kwargs_model, 'multi_band_list': multi_band_list,
                                'kwargs_special': kwargs_special}
