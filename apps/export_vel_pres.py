import pyvista
from pyvista import examples
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

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
x = np.linspace(xmin + delta / 2, xmax, nx)
y = np.linspace(ymin + delta / 2, ymax, ny)
z = np.linspace(zmin + delta / 2, zmax, nz)
grid_x, grid_y, grid_z = np.meshgrid(x, y, z)

# full-sized coordinates
x_full = np.linspace(-xmax, xmax, 2 * (nx))
y_full = np.linspace(-ymax, ymax, 2 * (ny))
z_full = np.linspace(zmin - delta, zmax, nz + 1)

# resolution for resampling
re_x = 120
re_y = 128
re_z = 128


def map_field_scalar(coord, var):
    # %% that takes some time now - mapping on a grid
    var_mapped = griddata((coord[:, 0], coord[:, 1], coord[:, 2]), var, (grid_x, grid_y, grid_z), method='nearest')

    # %% constructing full-sized 3d-array
    var_mapped_full = np.zeros([2 * (ny), 2 * (nx), nz + 1])

    # adding a cell layer of air below
    var_mapped_full[:, :, 0] = -1
    var_mapped_full[ny:, 0:nx, 1:] = np.flip(var_mapped, axis=1)
    var_mapped_full[ny:, nx:, 1:] = var_mapped
    var_mapped_full[0:ny, 0:nx, 1:] = np.flip(np.flip(var_mapped, axis=1), axis=0)
    var_mapped_full[0:ny:, nx:, 1:] = np.flip(var_mapped, axis=0)

    # removing ridges in the interpolated data
    for i in range(0, 30):
        var_mapped_full[i * 12 + 9:i * 12 + 9 + 6, :, 1:7] = -1

    # %% some checking here
    print('Dimensions of mapped variable: ', np.shape(var_mapped_full))

    return var_mapped_full


def map_field_u(coord, var):
    # %% that takes some time now - mapping on a grid
    var_mapped = griddata((coord[:, 0], coord[:, 1], coord[:, 2]), var, (grid_x, grid_y, grid_z), method='nearest')

    # %% constructing full-sized 3d-array
    var_mapped_full = np.zeros([2 * (ny), 2 * (nx), nz + 1])

    # adding a cell layer of air below
    var_mapped_full[:, :, 0] = -1
    # flip around y-axis -> need to invert u-velocity component
    var_mapped_full[ny:, 0:nx, 1:] = -np.flip(var_mapped, axis=1)
    var_mapped_full[ny:, nx:, 1:] = var_mapped
    # flip around y-axis -> need to invert u-velocity component
    var_mapped_full[0:ny, 0:nx, 1:] = np.flip(-np.flip(var_mapped, axis=1), axis=0)
    var_mapped_full[0:ny:, nx:, 1:] = np.flip(var_mapped, axis=0)

    # removing ridges in the interpolated data
    for i in range(0, 30):
        var_mapped_full[i * 12 + 9:i * 12 + 9 + 6, :, 1:7] = -1

    # %% some checking here
    print('Dimensions of mapped variable: ', np.shape(var_mapped_full))

    return var_mapped_full


def map_field_v(coord, var):
    # %% that takes some time now - mapping on a grid
    var_mapped = griddata((coord[:, 0], coord[:, 1], coord[:, 2]), var, (grid_x, grid_y, grid_z), method='nearest')

    # %% constructing full-sized 3d-array
    var_mapped_full = np.zeros([2 * (ny), 2 * (nx), nz + 1])

    # adding a cell layer of air below
    var_mapped_full[:, :, 0] = -1
    var_mapped_full[ny:, 0:nx, 1:] = np.flip(var_mapped, axis=1)
    var_mapped_full[ny:, nx:, 1:] = var_mapped
    # flip around x-axis -> need to invert v-velocity component
    var_mapped_full[0:ny, 0:nx, 1:] = -np.flip(np.flip(var_mapped, axis=1), axis=0)
    # flip around x-axis -> need to invert v-velocity component
    var_mapped_full[0:ny:, nx:, 1:] = -np.flip(var_mapped, axis=0)

    # removing ridges in the interpolated data
    for i in range(0, 30):
        var_mapped_full[i * 12 + 9:i * 12 + 9 + 6, :, 1:7] = -1

    # %% some checking here
    print('Dimensions of mapped variable: ', np.shape(var_mapped_full))

    return var_mapped_full


def resample_field(X, Y, Z, var, n_x=10, n_y=10, n_z=10):
    step_x = int(np.floor(np.shape(X)[0] / n_x))
    step_y = int(np.floor(np.shape(Y)[1] / n_y))
    step_z = int(np.floor(np.shape(Z)[2] / n_z))

    var_resampled = var[::step_x, ::step_y, ::step_z]
    print('Dimensions of resampled variable: ', np.shape(var_resampled))

    return var_resampled


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
    reader.set_active_time_value(float(file))

    # this is the real loading
    internal_mesh = reader.read()["internalMesh"]

    # get cell center coordinates
    coord = internal_mesh.cell_centers().points

    # get the phase indicator data
    c = internal_mesh.cell_data['C']
    # get the velocity data
    velocity = internal_mesh.cell_data['U']
    # get the pressure data
    P = internal_mesh.cell_data['p']

    u = velocity[:, 0]
    v = velocity[:, 1]
    w = velocity[:, 2]

    # consider mirroring for velocity fields in x and y coordinate (w is defined for the entire domain already)
    u_mapped = map_field_u(coord, u)
    v_mapped = map_field_v(coord, v)
    w_mapped = map_field_scalar(coord, w)
    p_mapped = map_field_scalar(coord, P)
    # c_mapped = map_field_scalar(coord, c)

    # %% creating new pyvista mesh with the new data
    X, Y, Z = np.meshgrid(x_full, y_full, z_full)

    x_resampled = resample_field(X, Y, Z, X, n_x=re_x, n_y=re_y, n_z=re_z)
    y_resampled = resample_field(X, Y, Z, Y, n_x=re_x, n_y=re_y, n_z=re_z)
    z_resampled = resample_field(X, Y, Z, Z, n_x=re_x, n_y=re_y, n_z=re_z)
    u_resampled = resample_field(X, Y, Z, u_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    v_resampled = resample_field(X, Y, Z, v_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    w_resampled = resample_field(X, Y, Z, w_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    p_resampled = resample_field(X, Y, Z, p_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    # c_resampled = resample_field(X, Y, Z, c_mapped, n_x=re_x, n_y=re_y, n_z=re_z)
    # w_resampled = 0
    # p_resampled = 0
    c_resampled = 0

    # plot_iso_surface(x_resampled, y_resampled, z_resampled, c_resampled)
    plot_iso_surface(x_resampled, y_resampled, z_resampled, u_resampled)
    plot_iso_surface(x_resampled, y_resampled, z_resampled, v_resampled)
    plot_iso_surface(x_resampled, y_resampled, z_resampled, w_resampled)
    # plot_iso_surface(x_resampled, y_resampled, z_resampled, p_resampled)
    # plot_velocity_field(x_resampled, y_resampled, z_resampled, u_resampled, v_resampled, w_resampled)

    # rotate to match PIFu coordinate system (x,y,z) -> (x,z,y)
    u_resampled = u_resampled.transpose((1, 2, 0))
    v_resampled = v_resampled.transpose((1, 2, 0))
    w_resampled = w_resampled.transpose((1, 2, 0))
    p_resampled = p_resampled.transpose((1, 2, 0))
    # c_resampled = c_resampled.transpose((1, 2, 0))

    print('Dimensions after transpose: ', np.shape(u_resampled))

    return x_resampled, y_resampled, z_resampled, u_resampled, v_resampled, w_resampled, p_resampled, c_resampled


pathToSimulation = "/net/istmtrinculo/volume2/data2/hi208/fh2_work/foam/vu3498-3.2/run/60um/6ZellenproRille_higher/"
out_dir = "/net/istmhome/users/hi227/Projects/PIFu-master/train_data_DFS2023C/"

reader = pyvista.POpenFOAMReader(pathToSimulation + "foam.foam")

dirnames = [name for name in os.listdir(pathToSimulation)]

dirnames.sort()

# only get folders starting with a number i.e. time steps
dirnames = [x for x in dirnames if x[0].isdigit()]
# folder 0 is empty -> skip
# folder 249, t=0.003824982 appears to be broken -> skip
dirnames.remove('0.003824982')
dirnames.remove('1.5e-05')
print(len(dirnames))
print(dirnames)
# dirnames = dirnames[50:]

for i, file in enumerate(dirnames):

    if i == 0:
        continue

    if file == "constant" or file == "system" or file == "foam.foam":
        continue

    iter_start_time = time.time()
    x, y, z, u, v, w, p, c = extractFromSimulation(file)
    exit()

    # saving data
    savedirvel = os.path.join(out_dir, 'VEL', str(i).zfill(4))
    if not os.path.exists(savedirvel):
        os.mkdir(savedirvel)

    savedirpres = os.path.join(out_dir, 'PRES', str(i).zfill(4))
    if not os.path.exists(savedirpres):
        os.mkdir(savedirpres)

    # np.save(os.path.join(savedirvel, "x_train.npy"), x)
    # np.save(os.path.join(savedirvel, "y_train.npy"), y)
    # np.save(os.path.join(savedirvel, "z_train.npy"), z)
    # np.save(os.path.join(savedirvel, "u_train.npy"), u)
    # np.save(os.path.join(savedirvel, "v_train.npy"), v)
    # np.save(os.path.join(savedirvel, "w_train.npy"), w)
    # np.save(os.path.join(savedirpres, "p_train.npy"), p)

    # np.save(os.path.join(savedirvel, "c_train.npy"), c)

    iter_end_time = time.time()
    print('time: ', iter_end_time - iter_start_time)
