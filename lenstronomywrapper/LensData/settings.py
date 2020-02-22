class DefaultDataSpecifics(object):

    def __init__(self, background_rms=0.2, exp_time=1000, window_size=3.5, deltaPix=0.05, fwhm=0.1,
                 psf_type='GAUSSIAN'):

        numPix = float(window_size)/deltaPix
        self.background_rms = background_rms
        self.exp_time = exp_time
        self.numPix = int(numPix)
        self.deltaPix = deltaPix
        self.fwhm = fwhm
        self.psf_type = psf_type
        self.kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    def __call__(self):

        if self.psf_type == 'GAUSSIAN':
            out_psf = {'psf_type': 'GAUSSIAN', 'fwhm': self.fwhm, 'pixel_size': self.deltaPix, 'truncation': 5}
        elif self.psf_type == 'None':
            out_psf = {}
        out_data = (self.numPix, self.deltaPix, self.exp_time, self.background_rms)
        return out_psf, out_data
