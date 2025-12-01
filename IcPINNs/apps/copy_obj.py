import os
import glob
import shutil
import fnmatch

### Input dirs
# .obj 3D-geometries
in_path = "../../Blender_droplet_shear_flow/obj/"
#in_path = "../../../Data/Fink2018_non_axis_symmetrical/export_obj/"
# texture required to run rendering (arbibrary)
tex_path = "../../PIFu-master/render/tex/droplet0_250_dif_2k.jpg"

### Ouput dirs
out_path = "../train_data_DSH2024A/GEO/OBJ/"


def copy_files(in_path,tex_source,subject_name):
    # set path for obj, prt
    mesh_path = os.path.join(in_path, subject_name)
    subject_path = os.path.join(subject_name[:-4])
    if not os.path.exists(mesh_path):
        print('ERROR: obj file does not exist!!', mesh_path)
        return

    # create new directory
    os.makedirs(os.path.join(out_path, subject_path, 'tex'),exist_ok=True)

    # move obj file
    cmd = 'cp %s %s' % (mesh_path, os.path.join(out_path, subject_path, subject_name))
    #cmd = 'mv %s %s' % (mesh_path, os.path.join(out_path, subject_path, subject_name))
    print(cmd)
    os.system(cmd)

    # copy tex file
    tex_name = os.path.join(subject_name[:-4] + '_dif_2k.jpg')
    shutil.copy(tex_source, os.path.join(out_path, subject_path, 'tex', tex_name))

    return


# Load in the images
from os import walk

filenames = []
for (dirpath, dirnames, file_name) in walk(in_path):
    filenames.extend(file_name)
    print(filenames)
    break

filenames.sort()

for fname in filenames:
    if fname.endswith('.obj'):
        print(fname)
        copy_files(in_path,tex_path, fname)






