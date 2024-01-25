#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
from os import walk
import glob
import shutil
import fnmatch
import random

### Split
ratio_train = 0.9
ratio_val = 1.0


### Input dirs
out_path = "../PIFu-master/train_data_DFS2023C/"
in_path = "../PIFu-master/train_data_DFS2023C/RENDER/"

def pick_dataset(dnames,out_path):

    droplet_ds = []
    structured_ds = []
    for name in dnames:
      if name.startswith('droplet'):
          droplet_ds.append(name)
      else:
          structured_ds.append(name)

    n = int((1 - ratio_train) * len(structured_ds))
    print("val samples, n=", n)
    val_test_names = random.sample(structured_ds, n)

    n_val = int(ratio_val * len(val_test_names))
    val_names = random.sample(val_test_names, n_val)
    test_names = droplet_ds

    valdir = os.path.join(out_path, 'val.txt')
    testdir = os.path.join(out_path, 'test.txt')

    # open file in write mode
    with open(valdir, 'w') as fp:
        for item in val_names:
            # write each item on a new line
            fp.write("%s\n" % item)
        print(val_names)
        print('val split Done')

    with open(testdir, 'w') as fp:
        for item in test_names:
            # write each item on a new line
            fp.write("%s\n" % item)
        print(test_names)
        print('test split Done')

    return


def pick_random(dnames,out_path):
    n = int((1-ratio_train) * len(dnames))
    val_test_names = random.sample(dnames, n)
    
    n_val = int(ratio_val * len(dnames))
    val_names = random.sample(val_test_names, n_val)
    test_names = list(set(val_test_names) - set(val_names))

    valdir = os.path.join(out_path, 'val.txt')
    testdir = os.path.join(out_path, 'test.txt')

    # open file in write mode
    with open(valdir, 'w') as fp:
        for item in val_names:
            # write each item on a new line
            fp.write("%s\n" % item)
        print(val_names)
        print('val split Done')

    with open(testdir, 'w') as fp:
        for item in test_names:
            # write each item on a new line
            fp.write("%s\n" % item)
        print(test_names)
        print('test split Done')

    print("val samples, n=", len(val_names))
    print("test samples, n=", len(test_names))

    return



# Load the directory list
subfolders = [f.name for f in os.scandir(in_path) if f.is_dir() ]
subfolders.sort()

pick_dataset(subfolders, out_path)





