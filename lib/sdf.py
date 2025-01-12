import numpy as np


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples=10000, time_step=None):
    num_pts = points.shape[1]
    preds = np.zeros((5, num_pts))
    num_batches = num_pts // num_samples
    for i in range(num_batches):
        preds[:, i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples], time_step)
    if num_pts % num_samples:
        preds[:, num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:], time_step)

    return preds


def eval_grid(coords, eval_func, num_samples=10000, time_step=None):
    resolution = coords.shape[1:4]
    #print('resolution:', resolution)
    coords = coords.reshape([3, -1])
    preds = batch_eval(coords, eval_func, num_samples=num_samples, time_step=time_step)
    sdf = preds[0].reshape(resolution)
    # only get prediction in liquid phase
    u = np.multiply(preds[1].reshape(resolution), sdf)
    v = np.multiply(preds[2].reshape(resolution), sdf)
    w = np.multiply(preds[3].reshape(resolution), sdf)
    p = np.multiply(preds[4].reshape(resolution), sdf)

    return sdf, u, v, w, p


def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=10000, time_step=None):
    resolution = coords.shape[1:4]
    #print('resolution:', resolution)
    ''' Modification for PINN to allow evaluation of all fields: alpha + u,v,w,p'''
    preds = np.zeros((5,) + resolution)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        preds[:, test_mask] = batch_eval(points, eval_func, num_samples=num_samples, time_step=time_step)
        dirty[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = preds[0, x, y, z]
                    v1 = preds[0, x, y, z + reso]
                    v2 = preds[0, x, y + reso, z]
                    v3 = preds[0, x, y + reso, z + reso]
                    v4 = preds[0, x + reso, y, z]
                    v5 = preds[0, x + reso, y, z + reso]
                    v6 = preds[0, x + reso, y + reso, z]
                    v7 = preds[0, x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # volume fraction in cell is all the same
                    if (v_max - v_min) < threshold:
                        preds[0, x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        preds[1, x:x + reso, y:y + reso, z:z + reso] = (preds[1, x, y, z] + preds[1, x + reso, y + reso, z + reso]) / 2
                        preds[2, x:x + reso, y:y + reso, z:z + reso] = (preds[2, x, y, z] + preds[2, x + reso, y + reso, z + reso]) / 2
                        preds[3, x:x + reso, y:y + reso, z:z + reso] = (preds[3, x, y, z] + preds[3, x + reso, y + reso, z + reso]) / 2
                        preds[4, x:x + reso, y:y + reso, z:z + reso] = (preds[4, x, y, z] + preds[4, x + reso, y + reso, z + reso]) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

        #print('preds: ', preds[0])
        sdf = preds[0].reshape(resolution)
        u = preds[1].reshape(resolution)
        v = preds[2].reshape(resolution)
        w = preds[3].reshape(resolution)
        p = preds[4].reshape(resolution)

    return sdf, u, v, w, p
