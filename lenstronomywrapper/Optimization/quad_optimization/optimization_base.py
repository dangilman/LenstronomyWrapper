class OptimizationBase(object):

    def __init__(self, lens_system):

        self.lens_system = lens_system

    def _return_results(self, source, kwargs_lens_final, lens_model_full, return_kwargs):

        self._update_lens_system(source, kwargs_lens_final, lens_model_full)

        return kwargs_lens_final, lens_model_full, return_kwargs

    def _check_routine(self, opt_routine, contrain_params):

        if opt_routine == 'fixed_powerlaw_shear':
            pass
        elif opt_routine == 'fixedshearpowerlaw':
            assert contrain_params is not None
            assert 'shear' in contrain_params.keys()

    def _update_lens_system(self, source_centroid, new_kwargs, lens_model_full):

        self.lens_system.update_source_centroid(source_centroid[0], source_centroid[1])

        self.lens_system.update_kwargs_macro(new_kwargs)

        self.lens_system.update_light_centroid(new_kwargs[0]['center_x'], new_kwargs[0]['center_y'])

        self.lens_system.update_background_quasar(source_centroid[0], source_centroid[1])
