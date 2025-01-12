import trimesh
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt



def sample_occupancy_points(mesh, B_MIN, B_MAX, sigma, num_occupancy=5000, num_uvwp=5000, num_residuals=10000):

    '''occupancy points'''
    surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * num_occupancy)
    sample_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)

    # add random points within image space
    length = B_MAX - B_MIN
    random_points = np.random.rand(num_occupancy // 4, 3) * length + B_MIN
    sample_points = np.concatenate([sample_points, random_points], 0)
    np.random.shuffle(sample_points)

    inside = mesh.contains(sample_points)
    inside_points = sample_points[inside]
    outside_points = sample_points[np.logical_not(inside)]

    nin = inside_points.shape[0]
    inside_points = inside_points[:num_occupancy // 2] if nin > num_occupancy // 2 else inside_points
    outside_points = outside_points[:num_occupancy // 2] if nin > num_occupancy // 2 else outside_points[:(num_occupancy - nin)]

    occupancy_points = np.concatenate([inside_points, outside_points], 0).T
    occupancy_labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
    # save_samples_truncted_prob('out.ply', samples.T, labels.T)


    '''uvwp points'''
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_uvwp // 2)
    sample_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)

    random_points = np.random.rand(num_uvwp // 2, 3) * length + B_MIN
    uvwp_points = np.concatenate([sample_points, random_points], 0).T
    np.random.shuffle(uvwp_points)


    '''residual points'''
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_residuals // 2)
    sample_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)

    random_points = np.random.rand(num_residuals // 2, 3) * length + B_MIN
    residual_points = np.concatenate([sample_points, random_points], 0).T
    np.random.shuffle(residual_points)

    #print('occupancy points: ', occupancy_points.shape)
    #print('uvwp points: ', uvwp_points.shape)
    #print('residual points: ', residual_points.shape)

    return occupancy_points, occupancy_labels, uvwp_points, residual_points


def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def save_samples_rgb(fname, points, rgb):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def debug_sampling_points(render_data, sample_data):

    orimg = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, :] * 255.0)
    rot = render_data['calib'][0, :3, :3]
    trans = render_data['calib'][0, :3, 3:4]

    inside_pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
    pts = 0.5 * (inside_pts.numpy().T + 1.0) * render_data['img'].size(2)
    img = orimg.copy()
    for p in pts:
        img = cv2.circle(img, (p[0], p[1]), 0, (0, 255, 0), -1)

    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    cv2.imwrite('inside.png', img)

    outside_pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] < 0.5])  # [3, N]
    pts = 0.5 * (outside_pts.numpy().T + 1.0) * render_data['img'].size(2)
    img = orimg.copy()
    for p in pts:
        img = cv2.circle(img, (p[0], p[1]), 0, (255, 0, 255), -1)

    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    cv2.imwrite('outside.png', img)
