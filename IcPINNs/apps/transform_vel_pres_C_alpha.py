import numpy as np
import os



def get_alpha_from_C(C):
    alpha = C / 2 + 0.5

    return alpha


data_dir = "/net/istmhome/users/hi227/Projects/PINN-PIFu/train_data_DFS2024D/VEL"

dirnames = [name for name in os.listdir(data_dir)]
dirnames = [x for x in dirnames if x.startswith('droplet')]

dirnames.sort()
print(dirnames)
print('number of files to process: ', len(dirnames))

for i, dir in enumerate(dirnames):
    filename = os.path.join(data_dir, dir, "c_train.npy")
    C = np.load(filename)
    alpha = get_alpha_from_C(C)
    np.save(filename, alpha)


