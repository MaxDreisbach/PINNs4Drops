import torch

def project_velocity_vector_field(labels_u, labels_w, calibrations):
    '''
    Compute the projections of the velocity vector magnitudes, the transformation of the vector origin is performed
    by point projection already
    :param labels_u: velocity components labels in x coordinate
    :param labels_w: velocity components labels in z coordinate
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :return: proj_labels_u,
    proj_labels_w: [BxN] Tensor of projected velocity labels
    '''
    rot_mat = calibrations[:, :3, :3] # [B, 3, 3]
    print('rotation matrix: ', rot_mat)
    #rot_mat[:, 1, 1] = abs(rot_mat[:, 1, 1])
    rot_mat = rot_mat / rot_mat[:, 1, 1]
    labels_v = torch.zeros_like(labels_u)
    labels = torch.stack((labels_u, labels_v, labels_w), dim=1) # [B, 3, N]
    print('rotation matrix: ', rot_mat)
    #print(labels)
    #print('shape of rotation tensor: ', rot.size())
    #print('shape of u-comp labels tensor: ', labels_u.size())
    #print('shape of labels tensor: ', labels.size())

    labels_proj = torch.matmul(rot_mat, labels) # [B, 3, N]
    proj_labels_u = labels_proj[:, 0, :] # [B, 1, N]
    proj_labels_w = labels_proj[:, 2, :]

    #print('shape of projected labels tensor: ', labels_proj.size())
    #print('projected labels: ', labels_proj)
    #print('u-comp labels: ', proj_labels_u)
    #print('w-comp labels: ', proj_labels_w)
    #print('shape of projected u-comp tensor: ', proj_labels_u.size())
    #exit()

    return proj_labels_u, proj_labels_w

