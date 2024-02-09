#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch
from os import walk

### Input dirs
in_path = "/net/istmhome/users/hi227/Data/17112023_water_PDMS/Preprocessed/17112023_water_PDMS_SMG3_5_C001H001S0001_preprocessed/"
impact_frame = 0
FPS = 7500

# Load in the time step paths
filenames = []
for (dirpath, dirname, file_name) in os.walk(in_path):
    filenames.extend(file_name)
    break

imagenames = []
for file_name in filenames:
    if file_name.endswith('.png'):
        if not file_name.endswith('mask.png'):
            imagenames.append(file_name)

imagenames.sort()
imagenames = imagenames[impact_frame:]
print(imagenames)
print('Writing time labels for ', len(imagenames), ' image files',)

for i, file_name in enumerate(imagenames):
    t = i/FPS

    txtname = file_name[:-4] + '_time.txt'
    txtpath = os.path.join(in_path, txtname)
    print(txtname,t)
    with open(txtpath, 'w') as outfile:
        outfile.write(str(t))

        





