#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch

### Input dirs
# .obj 3D-geometries
in_path = "../../../Data/27072022_simulation_drop_impact_water_PC/obj/"
# texture required to run rendering (arbibrary)
tex_path = "../render/tex/droplet0_250_dif_2k.jpg"
# synthetic renders from blender
render_path = "../../../Data/27072022_simulation_drop_impact_water_PC/export_august/denoised/"
# masks synthetic renders from blender
mask_path = "../../../Projects/SynthDropexport/Export_PIFu/Masks/"
### Ouput dirs
out_path = "../../../Projects/PIFu-master/render/droplet_dataset/"
train_path = "../train_data/RENDER/"
train_mask_path = "../train_data/MASK/"
GEO_path = "../train_data/GEO/OBJ/"

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def copy_render(render_path_name,subject_name,train_path):
    # set path
    subject_path = os.path.join(subject_name[:-4])
    print(subject_path[:5])

    if subject_path[:5] == "model":
        render_path = os.path.join(render_path_name, subject_path[16:])
    else:
        render_path = os.path.join(render_path_name, subject_path)


    img_search = os.path.join("Image" + '*')

    print(render_path)
    print(img_search)
    img_name = find(img_search, render_path)
    img_name = sorted(img_name)
    print(img_name)
    img_path = img_name[-1]
    print(img_path)


    if not os.path.exists(img_path):
        print('ERROR: render file does not exist!!', img_path)
        return

    for i in range(0,360,1):
        try:
            new_name = os.path.join(train_path, subject_path, '%d_0_00.png' %i)
            print(new_name)
            shutil.copy(img_path, new_name)
        except:
            print("An exception occurred")

    return

# Load in the images
from os import walk

filenames = []
for (dirpath, dirnames, file_name) in walk(in_path):
    filenames.extend(file_name)
    print(filenames)
    break

for fname in filenames:
    if fname.endswith('.obj'):
        print(fname)
        #copy_files(in_path,tex_path,fname)
        #copy_mask(mask_path, fname, train_mask_path)
        copy_render(render_path, fname,train_path)
        #copy_obj_to_train_data(in_path, fname)





