#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
import glob
import shutil
import fnmatch

### Input dirs
# .obj 3D-geometries
in_path = "../render/Fink2018_augmented_structured_and_flat/"
# texture required to run rendering (arbibrary)
tex_path = "../render/tex/droplet0_250_dif_2k.jpg"
# synthetic renders from blender
render_path = "../../../Data/Fink2018/export_rotation/denoised/"
# masks synthetic renders from blender
mask_path = "../../../Projects/SynthDropexport/Export_PIFu/Masks/"
### Ouput dirs
out_path = "../../../Projects/PIFu-master/render/train_data/"
train_path = "../train_data/RENDER/"
train_mask_path = "../train_data/MASK/"
GEO_path = "../train_data/GEO/OBJ/"

def copy_files(in_path,tex_source,subject_name):
    # set path for obj, prt
    mesh_path = os.path.join(in_path, subject_name)
    subject_path = os.path.join(subject_name[:-4] + '_OBJ')
    if not os.path.exists(mesh_path):
        print('ERROR: obj file does not exist!!', mesh_path)
        return

    # create new directory
    os.makedirs(os.path.join(out_path, subject_path, 'tex'),exist_ok=True)

    # copy obj file
    cmd = 'cp %s %s' % (mesh_path, os.path.join(out_path,subject_path, subject_name))
    print(cmd)
    os.system(cmd)

    # copy tex file
    tex_name = os.path.join(subject_name[:-4] + '_dif_2k.jpg')
    shutil.copy(tex_source, os.path.join(out_path, subject_path, 'tex', tex_name))

    return


def copy_obj_to_train_data(in_path, subject_name, fname):
    # set path for obj
    mesh_path = os.path.join(in_path, subject_name)
    subject_path = subject_name[:-4]
    if not os.path.exists(mesh_path):
        print('ERROR: obj file does not exist!!', mesh_path)
        return

    # create new directory
    dest_path = os.path.join(GEO_path, fname[:-4])
    if not os.path.exists(dest_path):
        print('ERROR: destination path does not exist!!', dest_path)
        return
    print("----------------------------------------------------")
    print("copy from")
    print(mesh_path)
    print("to")
    print(dest_path)

    # copy obj file
    cmd = 'cp %s %s' % (mesh_path, dest_path)
    print(cmd)
    os.system(cmd)

    return

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def copy_render(render_path,subject_name,train_path):
    # set path for obj, prt
    subject_path = os.path.join(subject_name[:-4])
    img_search = os.path.join(subject_path + '*')

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
        new_name = os.path.join(train_path, subject_path, '%d_0_00.png' %i)
        print(new_name)
        shutil.copy(img_path, new_name)

    return


def copy_mask(mask_path,subject_name,train_mask_path):
    # set path for obj, prt
    subject_path = os.path.join(subject_name[:-4])
    img_search = os.path.join(subject_path + '*')

    print(img_search)
    img_name = find(img_search, mask_path)
    img_name = sorted(img_name)
    print(img_name)
    img_path = img_name[-1]
    print(img_path)



    if not os.path.exists(img_path):
        print('ERROR: render file does not exist!!', img_path)
        return

    # copy render image

    for i in range(0,360,1):
        new_name = os.path.join(train_mask_path, subject_path, '%d_0_00.png' %i)
        print(new_name)
        shutil.copy(img_path, new_name)

    return

# Load in the images
from os import walk

filenames = []
for (dirpath, dirnames, file_name) in walk(in_path):
    filenames.extend(file_name)
    print(filenames)
    break

for fname in dirnames:
    f_name = os.path.join(fname, fname[:-4] + '.obj')
    if f_name.endswith('.obj'):
        print(f_name)
        #copy_files(in_path,tex_path,fname)
        #copy_mask(mask_path, fname, train_mask_path)
        #copy_render(render_path, fname,train_path)
        copy_obj_to_train_data(in_path, f_name, fname)





