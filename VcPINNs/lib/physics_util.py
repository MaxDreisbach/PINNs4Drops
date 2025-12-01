from skimage import measure
import numpy as np
import torch
import time
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
from .plotting import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compute_errors(res, labels, norm):
    diff = labels - res

    abs_error = torch.sum(torch.abs(diff)).item()
    squared_error = torch.sum(diff ** 2).item()
    label_abs = torch.sum(torch.abs(labels)).item()
    label_squared = torch.sum(labels ** 2).item()
    elements = labels.numel()

    return abs_error, squared_error, label_abs, label_squared, elements


def compute_global_errors(total_abs, total_sq, total_abs_lab, total_sq_lab, total_num, eps=1e-8):
    l1_mean = total_abs / total_num
    l2_mean = (total_sq / total_num) ** 0.5
    rel_l1 = total_abs / (total_abs_lab + eps)
    rel_l2 = (total_sq ** 0.5) / (total_sq_lab ** 0.5 + eps)
    return l1_mean, rel_l1, l2_mean, rel_l2


def get_integral_energy(e, res, rp):
    # we are dealing with very small numbers here -> multiply and later divide by 1000 ** 3 to avoid loss of precision from float underflow
    dV = (1 / res * rp * 2) ** 3 * 1000 ** 3
    #print("dx [m]:", (1 / res * rp * 2))
    #print("dV [m^3]:", (1 / res * rp * 2) ** 3)
    E = np.sum(e * dV) / 1000 ** 3
    return E


def calculate_energy_contributions(opt, coords, alpha, u, v, w, verts, faces, U_ref, L_ref, rho_1, rho_2, sigma, g, y_ground, contact_angle, plot_diagnostic=False):
    '''
    Calculation of kinetic, potential, and dissipated energy in droplet
    in dimensional form
    :param u,v,w: dimensional velocity components
    '''
    #L_ref = L_ref / 1.06784550647
    y_coord = coords[1, :, :, :]
    ground_level = y_ground - 1

    rho_M = alpha * rho_1 + (1 - alpha) * rho_2

    # [-1:1,-1:1,-1:1] domain -> apply L_ref
    y_dim = (y_coord + 1) * L_ref    
    y_ground_dim = y_ground * L_ref

    E_kin = rho_M / 2 * (u ** 2 + v ** 2 + w ** 2)
    E_pot = rho_M * np.abs(g) * (y_dim - y_ground_dim)
    
    # mask to only get prediction in droplet
    alpha_mask = np.round(alpha)
    e_kin = E_kin * alpha_mask
    e_pot = E_pot * alpha_mask
    
    y_drop = y_dim[alpha > 0.5]
    print("mean y_drop [mm]:", np.mean(y_drop) * 1000)
    print("y_ground_dim [mm]:", y_ground_dim * 1000)
    
    print("mean velocity:", np.mean(np.sqrt(u[alpha > 0.5]**2 + v[alpha > 0.5]**2 + w[alpha > 0.5]**2)))

    # Calculate surface areas
    surface_areas = calculate_mesh_surface_areas(verts, faces, ground_level, tolerance=1e-6)
    
    if plot_diagnostic:
        plot_triangles(verts, faces, surface_areas)

    # Print results
    print(
        f"Ground level area [mm^2]: {surface_areas['ground_level_area'] * L_ref ** 2 * 1000:.6f} ({surface_areas['statistics']['ground_percentage']:.2f}%)")
    print(
        f"Above ground area [mm^2]: {surface_areas['above_ground_area'] * L_ref ** 2 * 1000:.6f} ({surface_areas['statistics']['above_percentage']:.2f}%)")
    print(
        f"Mixed area (crossing ground) [mm^2]: {surface_areas['mixed_area'] * L_ref ** 2 * 1000:.6f} ({surface_areas['statistics']['mixed_percentage']:.2f}%)")

    # Ground contribution calculated from Young equation (sigma_ls - sigma_gs) = sigma cos(theta_eq) with theta_eq=102 for PDMS (69.5 for PLA)
    E_surf = (surface_areas['above_ground_area']) * L_ref ** 2 * sigma + \
             (surface_areas['ground_level_area'] + surface_areas['mixed_area']) * L_ref ** 2 * sigma * (
                 - np.cos(np.radians(contact_angle)))

    E_kin = get_integral_energy(e_kin, opt.resolution, L_ref)
    E_pot = get_integral_energy(e_pot, opt.resolution, L_ref)

    return E_surf, E_kin, E_pot


def calculate_triangle_area(v1, v2, v3):
    """
    Calculate the area of a triangle given three vertices using cross product.

    Args:
        v1, v2, v3: numpy arrays of shape (3,) representing triangle vertices

    Returns:
        float: Area of the triangle
    """
    # Calculate two edge vectors
    edge1 = v2 - v1
    edge2 = v3 - v1

    # Cross product gives a vector whose magnitude is twice the triangle area
    cross_product = np.cross(edge1, edge2)

    # For 3D vectors, cross product returns a 3D vector
    area = 0.5 * np.linalg.norm(cross_product)

    return area


def classify_triangle_ground_level(v1, v2, v3, ground_level=0.0, tolerance=1e-6):
    """
    Classify a triangle based on its position relative to ground level.

    Args:
        v1, v2, v3: numpy arrays of shape (3,) representing triangle vertices
        ground_level: float, y-coordinate of ground level (assuming y is vertical)
        tolerance: float, tolerance for considering vertices at ground level

    Returns:
        str: 'ground' if triangle is at/near ground level, 'above' if above ground, 'mixed' if crossing
    """
    # Assuming y-coordinate is the vertical axis (change index if different)
    y_coords = np.array([v1[1], v2[1], v3[1]])

    # Check if all vertices are within tolerance of ground level
    at_ground = np.abs(y_coords - ground_level) <= tolerance
    above_ground = y_coords > (ground_level + tolerance)
    below_ground = y_coords < (ground_level - tolerance)

    if np.all(at_ground):
        return 'ground'
    elif np.all(above_ground) or np.all(at_ground | above_ground):
        return 'above'
    else:
        return 'mixed'  # Triangle crosses ground level


def calculate_mesh_surface_areas(verts, faces, ground_level=0.0, tolerance=1e-6):
    """
    Calculate surface areas of mesh separated by ground level.

    Args:
        verts: numpy array of shape (N, 3) containing vertex coordinates
        faces: numpy array of shape (M, 3) containing face indices
        ground_level: float, y-coordinate of ground level
        tolerance: float, tolerance for considering vertices at ground level

    Returns:
        dict: Dictionary containing surface area breakdown
    """
    total_area = 0.0
    ground_area = 0.0
    above_ground_area = 0.0
    mixed_area = 0.0

    ground_faces = []
    above_faces = []
    mixed_faces = []

    for face_idx, face in enumerate(faces):
        # Get the three vertices of the triangle
        v1 = verts[face[0]]
        v2 = verts[face[1]]
        v3 = verts[face[2]]

        # Calculate triangle area
        area = calculate_triangle_area(v1, v2, v3)
        total_area += area

        # Classify triangle relative to ground level
        classification = classify_triangle_ground_level(v1, v2, v3, ground_level, tolerance)

        if classification == 'ground':
            ground_area += area
            ground_faces.append(face_idx)
        elif classification == 'above':
            above_ground_area += area
            above_faces.append(face_idx)
        else:  # mixed
            mixed_area += area
            mixed_faces.append(face_idx)

    return {
        'total_area': total_area,
        'ground_level_area': ground_area,
        'above_ground_area': above_ground_area,
        'mixed_area': mixed_area,  # Triangles that cross ground level
        'ground_faces': ground_faces,
        'above_faces': above_faces,
        'mixed_faces': mixed_faces,
        'statistics': {
            'ground_percentage': (ground_area / total_area) * 100 if total_area > 0 else 0,
            'above_percentage': (above_ground_area / total_area) * 100 if total_area > 0 else 0,
            'mixed_percentage': (mixed_area / total_area) * 100 if total_area > 0 else 0
        }
    }

def plot_triangles(verts, faces, results):
    """
    Plot triangles in different colors based on classification.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for each type
    colors = {'ground': 'red', 'above': 'green', 'mixed': 'blue'}
    
    # Plot each type of triangle
    for face_type, face_list in [('ground', results['ground_faces']), 
                                ('above', results['above_faces']), 
                                ('mixed', results['mixed_faces'])]:
        if face_list:
            triangles = [verts[faces[i]] for i in face_list]
            collection = Poly3DCollection(triangles, alpha=0.7, facecolor=colors[face_type])
            ax.add_collection3d(collection)
    
    # Set axis limits
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    
    plt.title('Triangle Classification')
    plt.show()