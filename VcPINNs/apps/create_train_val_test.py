import os
from os import walk
import glob
import shutil
import fnmatch
import random

### Split
ratio_train = 0.15
test_start = 469
test_end = 570


### Input dirs
out_path = "../train_data_DFS2025A/"
in_path = "../train_data_DFS2024A/RENDER/"

def pick_random(dnames,out_path):
    # spare out test section
    dnames.sort()
    test_names = dnames[test_start:test_end]
    
    train_val_names = list(set(dnames) - set(test_names))
    
    n = int(ratio_train * len(dnames))
    train_names = random.sample(train_val_names, n)
    val_names = list(set(train_val_names) - set(train_names))
    

    print("no. train samples: %s (%.2f%%) " % (len(train_names) , len(train_names)/len(dnames)*100))
    print("no. val samples: %s (%.2f%%) " % (len(val_names) , len(val_names)/len(dnames)*100))
    print("no. test samples: %s (%.2f%%) " % (len(test_names) , len(test_names)/len(dnames)*100))
    
    
    val_names.sort()
    test_names.sort()

    valdir = os.path.join(out_path, 'val.txt')
    testdir = os.path.join(out_path, 'test.txt')

    # open file in write mode
    with open(valdir, 'w') as fp:
        for item in val_names:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(testdir, 'w') as fp:
        for item in test_names:
            # write each item on a new line
            fp.write("%s\n" % item)

    return



# Load the directory list
subfolders = [f.name for f in os.scandir(in_path) if f.is_dir() ]

pick_random(subfolders, out_path)

print('train/val/test split Done')



