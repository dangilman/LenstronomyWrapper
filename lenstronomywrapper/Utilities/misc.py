import subprocess
import shutil
import numpy as np

def create_directory(dirname=''):

    proc = subprocess.Popen(['mkdir', dirname])
    proc.wait()

def delete_dir(dirname=''):

    shutil.rmtree(dirname)

def write_fluxes(filename, fluxes, mode='a'):

    if mode == 'append':
        m = 'a'
    else:
        m = 'w'

    if fluxes.ndim == 1:

        with open(filename, m) as f:

            for val in fluxes:
                f.write(str(val) + ' ')
            f.write('\n')
    else:

        N = int(np.shape(fluxes)[0])

        with open(filename,m) as f:
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

def write_macro(output_path, kwargs_macro, mode):

    param_names = ''
    for key in kwargs_macro[0].keys():
        param_names += key + ' '

    with open(output_path, mode) as f:

        for row in range(0, len(kwargs_macro)):

            if row == 0 and mode == 'w':
                f.write(param_names + '\n')

            for key in kwargs_macro[row].keys():
                f.write(str(np.round(kwargs_macro[row][key], 4)) + ' ')
            f.write('\n')

