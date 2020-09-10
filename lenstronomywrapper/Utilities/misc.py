import subprocess
import shutil
import numpy as np

def write_lensdata(filename, x_image, y_image, fluxes, tdelay,
                   source_x=0, source_y=0):

    assert len(x_image) == 4

    x_image = np.round(x_image, 4)
    y_image = np.round(y_image, 4)
    fluxes = np.round(fluxes, 5)
    tdelay = np.round(tdelay, 4)

    with open(filename, 'w') as f:
        line = '4 ' + str(source_x) + ' '+ str(source_y) + ' '
        for (xi, yi, fi, ti) in zip(x_image, y_image, fluxes, tdelay):
            line += str(xi) + ' ' + str(yi) + ' '+ \
                        str(fi) + ' '+str(ti) + ' '
        f.write(line)

def create_directory(dirname=''):

    proc = subprocess.Popen(['mkdir', dirname])
    proc.wait()

def delete_dir(dirname=''):

    shutil.rmtree(dirname)

def write_fluxes(filename, fluxes, mode='a'):

    if fluxes.ndim == 1:

        with open(filename, mode) as f:

            for val in fluxes:
                f.write(str(val) + ' ')
            f.write('\n')
    else:

        N = int(np.shape(fluxes)[0])

        with open(filename,mode) as f:
            for n in range(0,N):
                for val in fluxes[n,:]:
                    f.write(str(val)+' ')
                f.write('\n')

def write_params(params, fname, header, mode, write_header=False):

    with open(fname, mode) as f:

        if write_header:
            f.write(header+'\n')

        if np.shape(params)[0] == 1:
            print(params)
            for p in range(0, len(params)):
                f.write(str(float(params[p]))+' ')

        else:
            for r in range(0,np.shape(params)[0]):
                row = params[r,:]
                for p in range(0,len(row)):
                    f.write(str(float(row[p]))+' ')
                f.write('\n')

def write_macro(output_path, kwargs_macro, mode, write_header):

    param_names = ''
    for key in kwargs_macro[0].keys():
        param_names += key + ' '

    with open(output_path, mode) as f:

        for row in range(0, len(kwargs_macro)):

            if row == 0 and write_header:
                f.write(param_names + '\n')

            for key in kwargs_macro[row].keys():
                f.write(str(np.round(kwargs_macro[row][key], 4)) + ' ')
            f.write('\n')

def write_sampling_rate(output_path, rate):

    seconds_per_realization = 1/rate
    minutes_per_realization = np.round(seconds_per_realization / 60, 5)
    with open(output_path, 'a') as f:
        f.write(str(minutes_per_realization)+'\n')

