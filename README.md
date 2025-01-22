# PINNs4Drops: Convolutional feature-enhanced physics-informed neural networks for reconstructing two-phase flows

by Maximilian Dreisbach (Institute of Fluid Mechanics (ISTM) - Karlsruhe Institute of Technology (KIT))
and Elham Kiyani (Division of Applied Mathematics - Brown University)

This code allows the evalution and training of the physics-informed neural networks for spatio-temporal gas-liquid interface reconstruction in two-phase flows as presented 
in our article "Convolutional feature-enhanced physics-informed neural networks for reconstructing two-phase flows".

The datasets used in this work, as well as the weights of the trained PINNs are available on [KITopen](https://doi.org/10.35097/mmnxkbqqeye8p5tx)

If you have any questions regarding this code, please feel free to contact Maximilian (maximilian.dreisbach@kit.edu).

## Requirements
- Python 3 (required packages below)
- PyTorch
- json
- PIL
- skimage
- tqdm
- numpy
- cv2
- matplotlib

for training and data generation
- trimesh with pyembree
- pyexr
- PyOpenGL
- freeglut

## Tested for: 
(see requirements.txt)

## Getting Started
- Create conda environment from requirements.txt (`conda create --name <env> --file requirements.txt`)
- Download pre-processed glare-point shadowgraphy images from [KITopen](https://doi.org/10.35097/mmnxkbqqeye8p5tx)
- OR use processing scripts on own data (see [GPS-Processing GitHub](https://github.com/MaxDreisbach/GPS-Processing))
- Download network weights and move the file to `./PINNs4Drops/checkpoints/`
- OR train the network on new data (see below)
- Run eval.py for volumetric reconstruction (see below)
- Open .obj file of reconstructed interface in Meshlab, Blender, or any 3D visualization software 

## Evaluation
This script reconstructs each image in `path_to_processed_image_data` into an `.obj` file representing the 3D gas-liquid interface and saves it under `./PIFu/results/name_of_experiment`.

`python -m apps.eval --name {name of experiment} --test_folder_path {path to processed image data} --dataroot {path to dataset with flow_case.json} --load_netG_checkpoint_path {path to network weights}`




## Data Generation (Linux Only)
The data generation uses codes adapted from PIFu by Saito et al. (2019), see [PIFu GitHub](https://github.com/shunsukesaito/PIFu) for further reference.
The following code should be run with [pyembree](https://github.com/scopatz/pyembree), as it is otherwise very slow. \
The data generation requires `.obj` files of ground truth 3D gas-liquid interfaces, obtained e.g. by numerical simulation. 
First, binary masks, placeholder renderings, and calibration matrices are computed from the specified `.obj` files.
Then, physically-based rendering in Blender is used to generate realistic synthetic images resembling the recordings from the experiments.

1. The following script precomputes spherical harmonic coefficients for rendering. Adjust the path to the `.obj` files in `prt_util_batch.py`.
```
python -m apps.prt_util_batch
```
2. The following script creates renderings, masks, and calibration matrices representing the relative orientation of the renderings and 3D geometries. The files are saved in newly created folders named `GEO`, `RENDER`, `MASK`, `PARAM`, `UV_RENDER`, `UV_MASK`, `UV_NORMAL`, and `UV_POS` under the specified training data path. Adjust the path to the `.obj` files in `render_data_batch.py`.
```
python -m apps.render_data_batch
```
3. Run the synthetic training data generation in Blender (see [Render-GPS GitHub](https://github.com/MaxDreisbach/RenderGPS))
4. Copy the renderings from the Blender output folder into the `RENDER` folder

## Training (Linux Only)
The following code should be run with [pyembree](https://github.com/scopatz/pyembree), as it is otherwise very slow. 

Run the following script to train the reconstruction network. The intermediate checkpoints are saved under `./checkpoints`. You can add `--batch_size` and `--num_sample_input` flags to adjust the batch size and the number of sampled points based on available GPU memory. The flags `--random_scale` and `--random_trans` enable data augmentation and perform random scaling and random cropping with translation of the input images.
```
python -m apps.train_shape --dataroot {path_to_training_data} --random_scale --random_trans
```


# Training the network #
### From PIFuT environment:
python -m apps.train_shape --dataroot ../PIFu-master/train_data_DFS2023C --name DFS2024D-PINN --num_epoch 8 --batch_size 1 --no_gen_mesh --gpu_id 1 --w_vel 1 --w_pres 1 --w_conti 10 --w_phase 10 --w_nse 1 --n_vel_pres_data 5000

# Evaluation #
### Plot predicted fields
python -m apps.plot_predicted_fields --dataroot ../PIFu-master/train_data_DFS2023C --name DFS2024D-PINN --gpu_id 0 --load_netG_checkpoint_path ./checkpoints/DFS2024D-PINN/netG_latest --resolution 512 --num_gen_mesh_test 50 --num_sample_inout 500000 --n_vel_pres_data 500000

### Inference
python ./apps/eval.py --name 'DFS2024D-PINN_PDMS_0s2' --batch_size 1 --num_stack 4 --gpu_id 0 --num_hourglass 2 --resolution 512 --hg_down 'ave_pool' --norm 'group' --norm_color 'group' --test_folder_path '../PIFu-master/Processed/PDMS_0s2' --load_netG_checkpoint_path './checkpoints/DFS2024D-PINN/netG_epoch_3' --load_checkpoint_path './checkpoints/net_C'

### Calculate train and val errors (From PIFu4 environment)
python -m apps.eval_IOU --dataroot ../PIFu-master/train_data_DFS2023C --name DFS2023C-PINN --gpu_id 0 --load_netG_checkpoint_path ./checkpoints/DFS2024D-PINN/netG_epoch_3 --resume_epoch 3 --batch_size 100

python -m apps.eval_IOU --dataroot ./train_data_DF2022 --gpu_id 1 --load_netG_checkpoint_path ./checkpoints/netG_epoch_49 --resume_epoch 49

### Generate train and val meshes of alpha field
### From PIFu4 environment:
python -m apps.eval_IOU --dataroot ./train_data_DS2022 --gpu_id 1 --load_netG_checkpoint_path ./checkpoints/example/netG_epoch_4 --batch_size 1 --num_gen_mesh_test 100 --name DS2022_train_val --no_num_eval --num_sample_inout 1


## Related Research
- This code is based on "PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization" by Saito et al. (2019) \
(see https://github.com/shunsukesaito/PIFu)
- "Physics-informed neural networks for phase-field method in two-phase flow" Qiu et al. (2022)
- "Inferring incompressible two-phase flow fields from the interface motion using physics-informed neural networks" Buhendwa et al. (2021)




