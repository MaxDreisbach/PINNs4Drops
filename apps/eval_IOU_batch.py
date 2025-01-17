''' Usage: Run python script, that opens eval_IOU for each saved checkpoints
python -m apps.eval_IOU_batch -f 'DFS2024A' -d './train_data_DFS2023C/'
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
parser.add_argument('-f', '--file_dir', type=str, default= 'DFS2024D-PINN_17')
parser.add_argument('-d', '--data_folder', type=str, default='./train_data_DFS2024D')
parser.add_argument('-n', '--num_files', type=int, default=1000)
parser.add_argument('-g', '--GPU_ID', type=int, default=0)
parser.add_argument('-M', '--model', type=str, default= 'VOF')
parser.add_argument('-L', '--LAAF', type=bool, default= False)
args = parser.parse_args()

data_path = './checkpoints/' + args.file_dir

if args.model == 'CH':
  base_command = "python -m apps.eval_IOU_CH --no_gen_mesh"
elif args.model == 'CH2':
  base_command = "python -m apps.eval_IOU_CH2 --no_gen_mesh"
else:
  base_command = "python -m apps.eval_IOU --no_gen_mesh"

filelist = [file for file in sorted(os.listdir(data_path))]
epochs = np.linspace(1, 8, num=8)
print(epochs)


commands = []
for i,epoch in enumerate(epochs):

  if args.LAAF:
      command = base_command + ' --gpu_id ' + str(args.GPU_ID) + ' --batch_size ' + str(args.num_files) +' --dataroot ' + args.data_folder + ' --name ' + args.file_dir + ' --load_netG_checkpoint_path ./checkpoints/' + args.file_dir + '/netG_epoch_' + str(i) + ' --resume_epoch ' + str(i) + ' --RGB True ' + '--n_data 15000 '
  else:
      command = base_command + ' --gpu_id ' + str(args.GPU_ID) + ' --batch_size ' + str(args.num_files) +' --dataroot ' + args.data_folder + ' --name ' + args.file_dir + ' --load_netG_checkpoint_path ./checkpoints/' + args.file_dir + '/netG_epoch_' + str(i) + ' --resume_epoch ' + str(i) + ' --RGB True ' + '--n_data 15000 '
  
  print(command)
  commands.append(command)


processes = [Popen(cmd, shell=True) for cmd in commands]

for process in processes:
    process.wait()
