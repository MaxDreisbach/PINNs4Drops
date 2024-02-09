#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch
import time

### Input dirs
# synthetic renders from blender
in_path = "../train_data_DSH2024/GEO/OBJ/"
render_path = "../../Blender_droplet_shear_flow/denoised/"

### Ouput dirs
train_path = "../train_data_DSH2024/RENDER/"

### Starting mesh
START = 0

### Flag for dataset
IS_DROPLET = False

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def copy_render(render_path_name,subject_name,train_path,angle_step):
    # set path
    subject_path = os.path.join(subject_name[:-4])
    #print(subject_path[:5])

    if subject_path[:5] == "model":
        render_path = os.path.join(render_path_name, subject_path[16:])
    else:
        render_path = os.path.join(render_path_name, subject_path)

    for i in range(0,360,angle_step):
        #NEW
        rot_path = os.path.join(subject_path + '_%d' %i)
        if IS_DROPLET:
          rot_path = os.path.join(subject_path + '_%d' % 0)
        
        render_path = os.path.join(render_path_name, rot_path, 'Image0000.png')
        print(render_path)
        
        new_path = os.path.join(train_path, subject_path)
        
        if not os.path.exists(new_path):
        	os.mkdir(new_path)
        
        new_name = os.path.join(new_path, '%d_0_00.png' %i)
        print(new_name)
        
        if not os.path.exists(render_path):
            print('ERROR: render file does not exist!!', render_path)
            #exit()
            return

        try:
            shutil.copy(render_path, new_name)
        except:
            print("An exception occurred")
            exit()

        

    return

# Load in the images
from os import walk

filenames = []
for (dirpath, dirnames, fname) in walk(in_path):
    filenames.extend(fname)

filenames.sort()
filenames = filenames[START:]
#print(filenames)
print('Copying RGB-renderings for ', len(filenames)+START, ' object files',)

for count,filename in enumerate(filenames):
    if filename.endswith('.obj'):
        start_time = time.time()
        copy_render(render_path, filename, train_path, angle_step=10)
        print("--- {0} of {1}: {2:.2f} s to copy renders of {3} ---".format(count+START, len(filenames)+START, (time.time() - start_time), filename))




