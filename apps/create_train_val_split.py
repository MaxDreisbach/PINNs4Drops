#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

import os
from os import walk
import glob
import shutil
import fnmatch
import random

### Split
ratio_train = 0.15
ratio_val = 0.15


### Input dirs
out_path = "../train_data_DFS2024A/"
in_path = "../train_data_DFS2024A/RENDER/"

def pick_random(dnames,out_path):
    n = int((1-ratio_train) * len(dnames))
    print("val samples, n=")
    print(n)
    val_test_names = random.sample(dnames, n)
    
    n_val = int(ratio_val * len(dnames))
    val_names = random.sample(val_test_names, n_val)
    test_names = list(set(val_test_names) - set(val_names))
    
    val_names.sort()
    test_names.sort()

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

pick_random(subfolders, out_path)





