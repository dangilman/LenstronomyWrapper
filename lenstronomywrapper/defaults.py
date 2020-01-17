from pyhalo.Cosmology.cosmology import Cosmology

class CosmoDefaults(object):

    def __init__(self):

        self.default_cosmo = Cosmology()

cosmo_defaults = CosmoDefaults()
