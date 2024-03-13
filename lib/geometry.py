import torch


def index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have
    most values in the range of [-1, 1]. For example, values x = -1, y = -1 is the left-top pixel of input,
    and values x = 1, y = 1 is the right-bottom pixel of input.
    (https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    ''' y-axis needs to be flipped so that top left corner is [-1, -1] instead of [-1 1]'''
    u = uv[:, :, :, 0]
    v = -uv[:, :, :, 1]
    uv_flipped = torch.stack((u, v), dim=3)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv_flipped, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz

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
    rot = calibrations[:, :3, :3] # [B, 3, 3]
    rot = rot / rot[:, 1, 1]
    labels_v = torch.zeros_like(labels_u)
    labels = torch.stack((labels_u, labels_v, labels_w), dim=1) # [B, 3, N]
    #print('rotation matrix: ', rot)
    #print(labels)
    #print('shape of rotation tensor: ', rot.size())
    #print('shape of u-comp labels tensor: ', labels_u.size())
    #print('shape of labels tensor: ', labels.size())

    labels_proj = torch.matmul(rot, labels) # [B, 3, N]
    proj_labels_u = labels_proj[:, 0, :] # [B, 1, N]
    proj_labels_w = labels_proj[:, 2, :]

    #print('shape of projected labels tensor: ', labels_proj.size())
    #print('projected labels: ', labels_proj)
    #print('u-comp labels: ', proj_labels_u)
    #print('w-comp labels: ', proj_labels_w)
    #print('shape of projected u-comp tensor: ', proj_labels_u.size())
    #exit()

    return proj_labels_u, proj_labels_w



