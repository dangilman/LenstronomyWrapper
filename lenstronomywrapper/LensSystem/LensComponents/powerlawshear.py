from lenstronomywrapper.LensSystem.LensComponents.macromodel_base import ComponentBase
import numpy as np

class PowerLawShear(ComponentBase):

    def __init__(self, redshift, kwargs_init=None, convention_index=False,
                 reoptimize=False, prior=[], concentric_with_lens_model=None,
                 concentric_with_lens_light=None, kwargs_fixed=None,
                 custom_prior=None):

        """
        This class defines an ellipsoidal power law mass profile plus external shear
        one of the more commmon models used in lensing.

        :param redshift: the redshift of the mass profile
        :param kwargs_init: the key word arguments for lenstronomy
        Example:
        = [{power law profile}, {external shear}]
        = [{'theta_E': 1, 'center_x': 0., 'center_y':, 0., 'e1': 0.1, 'e2': -0.2, 'gamma': 2.03},
        {'gamma1': 0.04, 'gamma2': -0.02}]

        :param convention_index: sets the position convention to 'lensed' in True
        (see documentation in lenstronomy under 'observed_convention_index')
        :param reoptimize: whether to re-optimize this profile in extended source reconstructions
        :param prior: a prior to add when performing an extended source reconstruction with this mass model
        (see documentation in lenstronomy.sampling)
        """
        self.reoptimize = reoptimize
        self._prior = prior
        self._concentric_with_lens_model = concentric_with_lens_model
        self._concentric_with_lens_light = concentric_with_lens_light
        self.kwargs_fixed = kwargs_fixed

        super(PowerLawShear, self).__init__(self.lens_model_list, [redshift]*self.n_models,
                                            kwargs_init, convention_index, False, reoptimize,
                                            custom_prior)

    @property
    def concentric_with_lens_light(self):

        return self._concentric_with_lens_light

    @property
    def concentric_with_lens_model(self):

        return self._concentric_with_lens_model

    @property
    def priors(self):

        indexes = []
        priors = []
        for prior in self._prior:
            pname = prior[0]
            if pname == 'gamma1' or pname == 'gamma2':
                idx = 1
            else:
                idx = 0
            indexes.append(idx)
            priors.append(prior)

        return indexes, priors

    @property
    def n_models(self):
        return 2

    @property
    def fixed_models(self):

        if self.fixed:
            return self.kwargs
        else:
            if self.kwargs_fixed is None:
                return [{}, {'ra_0': 0, 'dec_0': 0}]
            else:
                return [self.kwargs_fixed, {'ra_0': 0, 'dec_0': 0}]

    @property
    def param_init(self):

        return self.kwargs

    @property
    def param_sigma(self):

        if self.reoptimize:

            theta_E_scale, gamma_scale, centroid_scale, e12_scale = 0.05, 0.1, 0.1, 0.1
            shear_scale = 0.2
            old_kwargs = self._kwargs[0]
            old_kwargs_shear = self._kwargs[1]
            new_kwargs = {}
            new_kwargs_shear = {}

            new_kwargs_shear['gamma1'] = max(0.001, np.absolute(shear_scale * old_kwargs_shear['gamma1']))
            new_kwargs_shear['gamma2'] = max(0.001, np.absolute(shear_scale * old_kwargs_shear['gamma2']))

            for key in self._kwargs[0].keys():
                if key == 'center_x' or key == 'center_y':
                    new_kwargs[key] = max(0.001, centroid_scale * old_kwargs[key])
                elif key == 'theta_E':
                    new_kwargs[key] = max(0.01, theta_E_scale * old_kwargs[key])
                elif key == 'gamma':
                    new_kwargs[key] = max(0.01, gamma_scale * old_kwargs[key])
                elif key == 'e1' or key == 'e2':
                    new_kwargs[key] = max(0.0005, np.absolute(e12_scale * old_kwargs[key]))
                else:
                    raise Exception('param name ' + str(key) + 'not recognized.')

            return [new_kwargs, new_kwargs_shear]
        else:
            return [{'theta_E': 0.25, 'center_x': 0.1, 'center_y': 0.1, 'e1': 0.2, 'e2': 0.2, 'gamma': 0.15},
                    {'gamma1': 0.1, 'gamma2': 0.1}]

    @property
    def param_lower(self):

        lower = [{'theta_E': 0.05, 'center_x': -2., 'center_y': -2., 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5},
                 {'gamma1': -0.3, 'gamma2': -0.3}]
        return lower

    @property
    def param_upper(self):

        upper = [{'theta_E': 4., 'center_x': 2., 'center_y': 2., 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5},
                 {'gamma1': 0.3, 'gamma2': 0.3}]
        return upper

    @property
    def lens_model_list(self):
        return ['EPL', 'SHEAR']

    @property
    def redshift_list(self):
        return [self.zlens] * len(self.lens_model_list)
