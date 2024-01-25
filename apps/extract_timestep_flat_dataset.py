#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch
from os import walk

### Input dirs
#in_path = "/net/istmtrinculo/volume2/data2/hi208/fh2_work/foam/vu3498-3.2/run/60um/6ZellenproRille_higher/"
in_path = "/net/istmtrinculo/volume2/data2/hi208/fh2_work/foam/vu3498-3.2/run/AllesGleichWieStrukturiert/"
GEO_path = "/net/istmhome/users/hi227/Projects/PIFu-master/train_data_DFS2023C/GEO/OBJ/"
### Ouput dirs
time_step_path = "/net/istmhome/users/hi227/Projects/PIFu-master/train_data_DFS2023C/TIME/"

# Load in the time step paths
dirnames = []
for (dirpath, dirname, file_name) in walk(in_path):
    dirnames.extend(dirname)
    break
    
dirnames.sort()
timestamps = [x for x in dirnames if x[0].isdigit()]
print(len(timestamps))

# get corresponding sample names from OBJ folder
objnames = []
for (dirpath, dirnames, file_name) in walk(GEO_path):
    objnames.extend(dirnames)
    break

from natsort import natsorted

objnames = natsorted(objnames)
samplenames = [x for x in objnames if x.startswith('droplet')]
print(len(samplenames))

for i, timestamp in enumerate(timestamps):
        savepath = os.path.join(time_step_path, samplenames[i])
        print(savepath, timestamp)
        # create new directory
        os.makedirs(savepath,exist_ok=True)

        txtname = os.path.join(savepath, 'time_step.txt')
        with open(txtname, 'w') as outfile:
            outfile.write(timestamp)

        





