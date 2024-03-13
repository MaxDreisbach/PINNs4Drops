#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch
from os import walk
from natsort import natsorted

### Input dirs
in_path = "/net/istmhome/users/hi227/Data/12_2023_Simulation_droplet_shear_flow/5_85mps_20p0muc/"
GEO_path = "/net/istmhome/users/hi227/Projects/PINN-PIFu/train_data_DSH2024/GEO//OBJ/"
### Ouput dirs
time_step_path = "/net/istmhome/users/hi227/Projects/PINN-PIFu/train_data_DSH2024/TIME/"

# Load in the time step paths
dirnames = []
for (dirpath, dirname, file_name) in walk(in_path):
    dirnames.extend(dirname)
    break
    
dirnames.sort()
dirnames = natsorted(dirnames)
timestamps = [x for x in dirnames if x[0].isdigit()]
print(len(timestamps))

# folder 0 is empty -> skip
timestamps.remove('0')

# get corresponding sample names from OBJ folder
objnames = []
for (dirpath, dirnames, file_name) in walk(GEO_path):
    objnames.extend(dirnames)
    break

objnames = natsorted(objnames)
samplenames = [x for x in objnames if x[0].isdigit()]
samplenames = samplenames[1:]


for i, timestamp in enumerate(timestamps):

    savepath = os.path.join(time_step_path, samplenames[i])
    print(savepath, timestamp)
    # create new directory
    os.makedirs(savepath,exist_ok=True)

    txtname = os.path.join(savepath, 'time_step.txt')
    with open(txtname, 'w') as outfile:
        outfile.write(str(timestamp))


        





