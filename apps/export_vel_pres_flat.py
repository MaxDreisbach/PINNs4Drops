import pyvista
from pyvista import examples
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import vtk
from numpy import genfromtxt
import pandas as pd
from natsort import natsorted

# %% define the domain geometry and resolution
nx = 201
ny = 180
nz = 500
delta = 1e-5

xmin = 0
xmax = 0.00201
ymin = 0
ymax = 0.0018
zmin = 0
zmax = 0.005

# creating grid to map to
x = np.linspace(-xmax, xmax, 2 * (nx))
y = np.linspace(-ymax, ymax, 2 * (ny))
z = np.linspace(zmin - delta, zmax, nz + 1)
grid_x, grid_y, grid_z = np.meshgrid(x, y, z)

# resolution for resampling
re_x = 120
re_y = 128
re_z = 128

def map_field(X, Y, Z, var):
    # %% that takes some time now - mapping on a grid
    var_mapped = griddata((X, Y, Z), var, (grid_x, grid_y, grid_z), method='nearest')

    # %% some checking here
    print('Dimensions of mapped variable: ', np.shape(var_mapped))

    return var_mapped


def resample_field(X, Y, Z, var, n_x=10, n_y=10, n_z=10):
    step_x = int(np.floor(np.shape(X)[0] / n_x))
    step_y = int(np.floor(np.shape(Y)[1] / n_y))
    step_z = int(np.floor(np.shape(Z)[2] / n_z))

    var_resampled = var[::step_x, ::step_y, ::step_z]
    print('Dimensions of resampled variable: ', np.shape(var_resampled))

    return var_resampled


def project_velocity_vector_field(X, Y, Z, U, V):
    ''' rotate vector field around z-axis'''

    U[X < 0] = -U[X < 0]
    V[Y < 0] = -V[Y < 0]
    W = np.zeros_like(U)
    # determine angle from X, Y coordinates
    phi = np.arctan(Y / (X+np.finfo(float).eps))

    # generate rotation matrix around z-axis
    rot = np.array([[np.cos(phi), -np.sin(phi), np.zeros_like(phi)], [np.sin(phi), np.cos(phi), np.zeros_like(phi)], [np.zeros_like(phi), np.zeros_like(phi), np.ones_like(phi)]]) # [N, 3, 3]
    rot = np.swapaxes(rot, 0, 2)
    vectors = np.stack([U, V, W], axis=1)  # [N, 3]

    # determine magnitude of U_proj and V_proj from U (V is negligibly small)
    vectors_proj = np.matmul(rot, vectors[:, :, None]).squeeze(-1)  # [N, 3]
    U_proj = vectors_proj[:, 0] # [B, 1, N]
    V_proj = -vectors_proj[:, 1]  # [B, 1, N]
    print('U min: ', np.min(U_proj), 'U max: ', np.max(U_proj))
    print('V min: ', np.min(V_proj), 'V max: ', np.max(V_proj))

    return U_proj, V_proj


def plot_velocity_field(X, Y, Z, U, V, W):
    print('Plotting velocity field of shape: ', np.shape(X))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.3, length=0.00001)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    plt.show()


def plot_iso_surface(X, Y, Z, var_mapped):
    mesh = pyvista.StructuredGrid(X, Y, Z)
    mesh.point_data['values'] = var_mapped.ravel(order='F')

    vmin = var_mapped.min()
    vmax = var_mapped.max()
    labels = dict(zlabel='Z', xlabel='X', ylabel='Y')
    contours = mesh.contour(np.linspace(vmin, vmax, 10))

    p = pyvista.Plotter()
    p.add_mesh(mesh.outline(), color="k")
    p.add_mesh(contours, opacity=0.25, clim=[vmin, vmax])
    p.show_grid(**labels)
    p.add_axes(**labels)
    p.show()


def extractFromSimulation(file):
    print('doing ' + str(file))
    csv_name = os.path.join(pathToSimulation, file)

    df = pd.read_csv(csv_name)
    X = df['Points:0'].to_numpy()
    Y = df['Points:1'].to_numpy()
    Z = df['Points:2'].to_numpy()

    C = df['C'].to_numpy()
    U = df['U:0'].to_numpy()
    V = df['U:1'].to_numpy()
    W = df['U:2'].to_numpy()
    P = df['p'].to_numpy()

    # project rotated U vector field to U and V
    U, V = project_velocity_vector_field(X, Y, Z, U, V)

    c_mapped = map_field(X, Y, Z, C)
    u_mapped = map_field(X, Y, Z, U)
    v_mapped = map_field(X, Y, Z, V)
    w_mapped = map_field(X, Y, Z, W)
    p_mapped = map_field(X, Y, Z, P)

    # %% creating new mesh with the new data
    X, Y, Z = np.meshgrid(x, y, z)

    x_resampled = resample_field(X, Y, Z, X, n_x=re_x, n_y=re_y, n_z=re_z)
    y_resampled = resample_field(X, Y, Z, Y, n_x=re_x, n_y=re_y, n_z=re_z)
    z_resampled = resample_field(X, Y, Z, Z, n_x=re_x, n_y=re_y, n_z=re_z)
    c_resampled = resample_field(X, Y, Z, c_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    u_resampled = resample_field(X, Y, Z, u_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    v_resampled = resample_field(X, Y, Z, v_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    w_resampled = resample_field(X, Y, Z, w_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    p_resampled = resample_field(X, Y, Z, p_mapped, n_x=re_x, n_y=re_y, n_z=re_z)

    #plot_iso_surface(x_resampled, y_resampled, z_resampled, u_resampled)
    #plot_iso_surface(x_resampled, y_resampled, z_resampled, v_resampled)
    #plot_iso_surface(x_resampled, y_resampled, z_resampled, w_resampled)
    #plot_iso_surface(x_resampled, y_resampled, z_resampled, p_resampled)

    # rotate to match PIFu coordinate system (x,y,z) -> (x,z,y)
    c_resampled = c_resampled.transpose((1, 2, 0))
    u_resampled = u_resampled.transpose((1, 2, 0))
    v_resampled = v_resampled.transpose((1, 2, 0))
    w_resampled = w_resampled.transpose((1, 2, 0))
    p_resampled = p_resampled.transpose((1, 2, 0))

    return x_resampled, y_resampled, z_resampled, c_resampled, u_resampled, v_resampled, w_resampled, p_resampled


pathToSimulation = "/net/istmhome/users/hi227/Data/Simulation_data/"
out_dir = "/net/istmhome/users/hi227/Projects/PINN-PIFu/train_data_DFS2024D/"

dirnames = [name for name in os.listdir(pathToSimulation)]
dirnames = [x for x in dirnames if x.endswith('.csv')]

dirnames.sort()
print(dirnames)
print('number of files to process: ', len(dirnames))

out_names = [name for name in os.listdir(os.path.join(out_dir, 'RENDER/'))]
out_names = [x for x in out_names if x.startswith('droplet')]
out_names = natsorted(out_names)
print(out_names)
print('number of files to process: ', len(out_names))

for i, file in enumerate(dirnames):

    iter_start_time = time.time()
    x_dat, y_dat, z_dat, c_dat, u_dat, v_dat, w_dat, p_dat = extractFromSimulation(file)

    # saving data
    savedirvel = os.path.join(out_dir, 'VEL', out_names[i])
    if not os.path.exists(savedirvel):
        os.mkdir(savedirvel)

    savedirpres = os.path.join(out_dir, 'PRES', out_names[i])
    if not os.path.exists(savedirpres):
        os.mkdir(savedirpres)

    np.save(os.path.join(savedirvel, "c_train.npy"), c_dat)
    np.save(os.path.join(savedirvel, "u_train.npy"), u_dat)
    np.save(os.path.join(savedirvel, "v_train.npy"), v_dat)
    np.save(os.path.join(savedirvel, "w_train.npy"), w_dat)
    np.save(os.path.join(savedirpres, "p_train.npy"), p_dat)

    iter_end_time = time.time()
    print('time: ', iter_end_time - iter_start_time)
