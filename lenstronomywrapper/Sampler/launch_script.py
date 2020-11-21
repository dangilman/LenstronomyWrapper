from lenstronomywrapper.Sampler.run import run
from lenstronomywrapper.Sampler.run_two_sources import run_two_sources
import sys
import os
from time import time

#job_index = int(sys.argv[1])
job_index = 1
# the name of the folder containing paramdictionary files
chain_ID = 'B2045_test_save_best'
# where to generate output files
#out_path = '/scratch/abenson/'
out_path = os.getenv('HOME') + '/data/sims/'

# wherever you put the launch folder containing the
# paramdictionary files
#paramdictionary_folder_path = '/scratch/abenson/'
paramdictionary_folder_path = os.getenv('HOME') + '/data/'

print(job_index)
t0 = time()
# launch and forget
test_mode = True
run(job_index, chain_ID, out_path,
    paramdictionary_folder_path, test_mode=test_mode)
tend = time()
print('time ellapsed: ', tend - t0)
