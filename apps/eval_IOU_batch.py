''' Usage: Run python script, that opens eval_IOU for each saved checkpoints
python P1_drop_render_start_subprocess.py -n 4 -s "yes"
'''

from operator import index
import os
import os.path
import sys
import subprocess
from subprocess import Popen
import argparse
import numpy as np
import math


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, default= 'DFS2023M')
parser.add_argument('-d', '--data_folder', type=str, default='train_data_DFS2023C')
parser.add_argument('-n', '--num_files', type=int, default=1000)
args = parser.parse_args()

data_path = '../checkpoints/' + args.file_dir
base_command = "python -m eval_IOU --gpu_id 1 --no_gen_mesh"

filelist = [file for file in sorted(os.listdir(data_path))]
print(filelist)


commands = []
for i,file in enumerate(filelist):
  command = base_command + ' --batch_size ' + str(args.num_files) +' --dataroot ../' + args.data_folder + ' --name ' + args.file_dir + ' --load_netG_checkpoint_path ../checkpoints/' + args.file_dir + '/netG_epoch_' + str(i) + ' --resume_epoch ' + str(i)
  print(command)
  commands.append(command)


processes = [Popen(cmd, shell=True) for cmd in commands]

for process in processes:
    process.wait()
