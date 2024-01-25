#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch
from os import walk

### Input dirs
in_path = "/net/istmhome/users/hi227/Projects/PIFu-master/Processed/PDMS_0s2/"
impact_frame = 0
FPS = 7500

# Load in the time step paths
filenames = []
for (dirpath, dirname, file_name) in walk(in_path):
    filenames.extend(file_name)
    break

filenames.sort()
filenames = filenames[impact_frame:]
#print(filenames)
print('Writing time labels for ', len(filenames), ' image files',)

for i, file_name in enumerate(filenames):
    if not file_name.endswith('mask.png'):
        t = i/FPS

        txtname = file_name[:-4] + '_time.txt'
        txtpath = os.path.join(in_path, txtname)
        print(txtname,t)
        with open(txtpath, 'w') as outfile:
            outfile.write(str(t))

        





