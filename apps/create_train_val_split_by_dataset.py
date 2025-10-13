#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
from os import walk
import glob
import shutil
import fnmatch
import random

### Split
ratio_train = 0.7
ratio_val = 0.2


### Input dirs
out_path = "../train_data_DFS2025A/"
in_path = "../train_data_DFS2024A/RENDER/"

def pick_dataset(dnames,out_path):

    droplet_ds = []
    structured_ds = []
    for name in dnames:
      if name.startswith('droplet'):
          droplet_ds.append(name)
      else:
          structured_ds.append(name)

    n = int((1 - ratio_train) * len(structured_ds))
    val_test_names = random.sample(structured_ds, n)

    n_val = int(ratio_val * len(val_test_names))
    val_names = random.sample(val_test_names, n_val)
    test_names_struct = list(set(val_test_names) - set(val_names))
    test_names  = list(set(test_names_struct) | set(droplet_ds))
    
    print("test samples, n=", len(test_names_struct))
    print("val samples, n=", len(val_names))
    
    train_names = sorted(list(set(structured_ds) - set(val_test_names)))
    print("train samples, n=", len(train_names))
    print('train samples')
    print(train_names)
    exit()

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




# Load the directory list
subfolders = [f.name for f in os.scandir(in_path) if f.is_dir() ]
subfolders.sort()

pick_dataset(subfolders, out_path)





