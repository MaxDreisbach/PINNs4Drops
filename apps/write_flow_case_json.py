import json
import os

''' Flow case parameters (from Simulation Fink et al. "Drop bouncing by micro-grooves" (2018)
 https://doi.org/10.1016/j.ijheatfluidflow.2018.02.014'''

# fluid parameters
U_0 = 0.62  # impact velocity
D_0 = 2.1 / 10 ** 3  # Droplet diameter
rp = 256 / 93.809 / 10 ** 3  # synthetic image reproduction scale [image domain - PINN domain]
sigma = 0.071  # surface tension
rho_1 = 998.2  # density of inside medium (water)
rho_2 = 1.204  # density of outside medium (air)
mu_1 = 1.0016 / 10 ** 3  # viscosity of inside medium (water)
mu_2 = 1.825 / 10 ** 5  # viscosity of outside medium (air)
g = 9.81  # gravity

# Simulation domain
xmin = - 0.00201 * 50000
xmax = 0.00201 * 50000
ymin = -1e-5 * 50000 - 2.75
ymax = 0.005 * 50000 - 2.75
yground = 1e-5 * 6 * 50000 - 2.75  # 60um from y0
zmin = - 0.0018 * 50000
zmax = 0.0018 * 50000
x_res = 134
y_res = 167
z_res = 120

# variable ranges for min-max normalization
x_norm_min = 550.0
x_norm_max = -550.0
t_norm_min = 0
t_norm_max = 70.0
u_norm_min = - 5.0
u_norm_max = 5.0
v_norm_min = -2.0
v_norm_max = 4.5
w_norm_min = - 5.0
w_norm_max = 5.0
p_norm_min = -1.25
p_norm_max = 4.0



# Data to be written
dictionary = {
    "U_0": U_0,
    "D_0": D_0,
    "rp": rp,
    "sigma": sigma,
    "rho_1": rho_1,
    "rho_2": rho_2,
    "mu_1": mu_1,
    "mu_2": mu_2,
    "g": g,
    "x_min": xmin,
    "x_max": xmax,
    "y_min": ymin,
    "y_max": ymax,
    "y_ground": yground,
    "z_min": zmin,
    "z_max": zmax,
    "x_res": x_res,
    "y_res": y_res,
    "z_res": z_res,
    "x_norm_min": x_norm_min,
    "x_norm_max": x_norm_max,
    "t_norm_min": t_norm_min,
    "t_norm_max": t_norm_max,
    "u_norm_min": u_norm_min,
    "u_norm_max": u_norm_max,
    "v_norm_min": v_norm_min,
    "v_norm_max": v_norm_max,
    "w_norm_min": w_norm_min,
    "w_norm_max": w_norm_max,
    "p_norm_min": p_norm_min,
    "p_norm_max": p_norm_max,
}

print(dictionary)

# Serializing json
json_object = json.dumps(dictionary, indent=4)

train_data_dir = './train_data_DFS2024D'

# Writing to sample.json
with open(os.path.join(train_data_dir, "flow_case.json"), "w") as outfile:
    outfile.write(json_object)


