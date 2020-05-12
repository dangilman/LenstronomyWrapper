from lenstronomywrapper.Sampler.run import run
import sys
import os

job_index = int(sys.argv[1])

# the name of the folder containing paramdictionary files
chain_ID = 'test_submit'

# where to generate output files
#out_path = '/scratch/abenson/'
out_path = os.getenv('HOME') + '/data/sims/'

# wherever you put the launch folder containing the
# paramdictionary files
#paramdictionary_folder_path = '/scratch/abenson/'
paramdictionary_folder_path = os.getenv('HOME') + '/data/'

print(job_index)
# launch and forget
run(job_index, chain_ID, out_path,
    paramdictionary_folder_path, True)



