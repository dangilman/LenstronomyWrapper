import numpy as np

class ExtendedSource(object):

    def __init__(self, minimum_image_sep, grid_resolution,
                 source_types, source_kwargs=[]):

        self.grid_resolution = grid_resolution

        self.is_point_source = []

        for (source, kwargs_source) in zip(source_types, source_kwargs):
            if source == 'GAUSSIAN':
                from lenstronomywrapper.ExtendedSource.source_models import GAUSSIAN
                self.is_point_source.append(False)
                self.sources.append(GAUSSIAN(**kwargs_source))
                self.grid_rmax, self.res = self._grid_rmax(source_kwargs['source_size'], grid_resolution)

            elif source == 'POINT':
                self.is_point_source.append(True)
                self.sources.append(None)
            else:
                raise ValueError('other source models not yet implemented')

        self.grid = []

        if minimum_image_sep is not None:
            for j in range(0, len(minimum_image_sep[0])):
                sep = minimum_image_sep[0][j]
                theta = minimum_image_sep[1][j]
                L = 0.5 * sep
                self.grid.append(RayShootingGrid(min(self.grid_rmax, L), self.res, rot=theta))

    def magnification(self, xpos, ypos, lensModel, kwargs_lens):

        flux = np.zeros_like(xpos)
        xgrids, ygrids = self._get_grids(xpos, ypos)
        beta_x_list = []
        beta_y_list = []
        for k, source in enumerate(self.sources):
            if self.is_point_source[k]:
                flux += lensModel.magnification(xpos, ypos, kwargs_lens)
            else:

                for i in range(0,len(xpos)):
                    if k==0:
                        bx, by = lensModel.ray_shooting(xgrids[i].ravel(), ygrids[i].ravel(), kwargs_lens)
                        beta_x_list.append(bx)
                        beta_y_list.append(by)

                    surface_brightness_image = source(beta_x_list[i], beta_y_list[i])
                    flux[i] += np.sum(surface_brightness_image * self.grid_resolution ** 2)

                    # n = int(np.sqrt(len(image)))
                    # print('npixels: ' , n)
                    # plt.imshow(image.reshape(n,n)); plt.show()
                    # a=input('continue')
                    #blended = flux_at_edge(image.reshape(n,n))
                    #blended = False
                    #if blended:
                    #    flux.append(np.nan)
                    #else:

                    #plt.imshow(image.reshape(n,n))
                    #plt.show()
                    #a=input('continue')

        return np.array(flux)

    def _get_grids(self, xpos, ypos):

        xgrid, ygrid = [], []

        for i, (xi, yi) in enumerate(zip(xpos, ypos)):

            xg, yg = self.grid[i].grid_at_xy(xi, yi)

            xgrid.append(xg)
            ygrid.append(yg)

        return xgrid, ygrid

    def _grid_rmax(self, size_asec, res):

        if size_asec < 0.0002:
            s = 0.005
        elif size_asec < 0.0005:
            s = 0.03
        elif size_asec < 0.001:
            s = 0.08
        elif size_asec < 0.002:
            s = 0.2
        elif size_asec < 0.003:
            s = 0.28
        elif size_asec < 0.005:
            s = 0.35

        else:
            s = 0.48

        return s,res

class RayShootingGrid(object):

    def __init__(self, side_length, grid_res, rot):

        N = 2*side_length*grid_res**-1

        self.x_grid_0, self.y_grid_0 = np.meshgrid(
            np.linspace(-side_length+grid_res, side_length-grid_res, N),
            np.linspace(-side_length+grid_res, side_length-grid_res, N))

        self.radius = side_length

        self._rot = rot

    @property
    def grid_at_xy_unshifted(self):
        return self.x_grid_0, self.y_grid_0

    def grid_at_xy(self, xloc, yloc):

        theta = self._rot

        cos_phi, sin_phi = np.cos(theta), np.sin(theta)

        gridx0, gridy0 = self.grid_at_xy_unshifted

        _xgrid, _ygrid = (cos_phi * gridx0 + sin_phi * gridy0), (-sin_phi * gridx0 + cos_phi * gridy0)
        xgrid, ygrid = _xgrid + xloc, _ygrid + yloc

        xgrid, ygrid = xgrid.ravel(), ygrid.ravel()

        return xgrid, ygrid
