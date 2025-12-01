#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch
import time

### Input dirs
# synthetic renders from blender
in_path = "../../../Projects/PIFu-master/train_data/GEO/OBJ/"
render_path = "../../../Projects/3D_droplet_augmentation/data/export_render_rotation/denoised/"

### Ouput dirs
train_path = "../train_data/RENDER/"

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
        render_path = os.path.join(render_path_name, rot_path, 'Image0000.png')
        #print(render_path)

        try:
            new_name = os.path.join(train_path, subject_path, '%d_0_00.png' %i)
            #print(new_name)
            #TODO: load image here and transform blue channel to BW image (with 3 channels)

            shutil.copy(render_path, new_name)
        except:
            print("An exception occurred")

        if not os.path.exists(render_path):
            print('ERROR: render file does not exist!!', render_path)
            exit()
            return

    return

# Load in the images
from os import walk

filenames = []
for (dirpath, dirnames, fname) in walk(in_path):
    filenames.extend(fname)

filenames.sort()
filenames = filenames[:100]
print(filenames)
print('Copying RGB-renderings for ', len(filenames), ' object files',)

for filename in filenames:
    if filename.endswith('.obj'):
        start_time = time.time()
        copy_render(render_path, filename, train_path, angle_step=10)
        print("--- %s seconds to copy rendered images for %s ---" % ((time.time() - start_time), filename))





