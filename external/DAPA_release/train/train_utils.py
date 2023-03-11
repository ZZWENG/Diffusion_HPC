import numpy as np
import torch
from torchvision.transforms import Normalize
from torchgeometry import rotation_matrix_to_angle_axis
import constants
from utils.geometry import perspective_projection
from train.render_utils import uv_renderer

normalize_T = Normalize(mean=constants.IMG_NORM_MEAN,
                        std=constants.IMG_NORM_STD)


def synthetic_gt(g_rotmat, g_pose, g_shape, g_cam, pred_rotmat,
                 smpl_model, options, use_opt_global_orient, texture_pred=None, backgrounds=None, sample_index=None):
    batch_size = g_rotmat.shape[0]
    device = g_rotmat.device  # cuda

    g_shape = g_shape.detach()
    new_gt = {}

    new_gt['betas'] = g_shape
    new_gt['pose'] = g_pose  # (B, 72)

    if use_opt_global_orient:
        pred_rotmat_hom = torch.cat(
            [pred_rotmat.view(-1, 3, 3),
             torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(
                 1, 3, 1).expand(batch_size*24, -1, -1)
             ], dim=-1)
        pred_pose = rotation_matrix_to_angle_axis(
            pred_rotmat_hom).contiguous().view(batch_size, -1)
        pred_pose[torch.isnan(pred_pose)] = 0.0
        new_gt['pose'][:, :3] = pred_pose[:, :3]
        new_out = smpl_model(betas=g_shape,
                             body_pose=g_rotmat[:, 1:],
                             global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
    else:
        new_out = smpl_model(betas=g_shape,
                             body_pose=g_rotmat[:, 1:],
                             global_orient=g_rotmat[:, 0].unsqueeze(1), pose2rot=False)

    new_model_joints = new_out.joints
    new_vertices = new_out.vertices

    # for seedlings and PW3D, we use openpose 25 keypoints
    S25 = torch.zeros([batch_size, 25, 4])
    S25[:, :, -1] = 1
    S25[:, :, :-1] = new_model_joints[:, :25, :]
    new_gt['pose_3d'] = S25.type(torch.float32)

    pred_cam_t = torch.stack([
        g_cam[:, 1], g_cam[:, 2],
        2*constants.FOCAL_LENGTH/(options.img_res * g_cam[:, 0] + 1e-9)
    ], dim=-1)
    camera_center = torch.zeros(batch_size, 2, device=device)

    keypoints_2d = perspective_projection(
        new_model_joints, translation=pred_cam_t,
        rotation=torch.eye(3).unsqueeze(0).expand(
            batch_size, -1, -1).to(device),
        focal_length=constants.FOCAL_LENGTH, camera_center=camera_center)
    keypoints_2d = keypoints_2d / (options.img_res / 2.)

    keypoints_2d = torch.cat([keypoints_2d, torch.ones(
        (batch_size, keypoints_2d.shape[1], 1)).to(device)], 2)
    new_gt['keypoints'] = keypoints_2d  # has grad info
    new_gt['has_smpl'] = torch.ones(batch_size)
    new_gt['has_pose_3d'] = torch.ones(batch_size)
    new_gt['is_flipped'] = torch.zeros(batch_size)
    new_gt['rot_angle'] = torch.zeros(batch_size)
    new_gt['dataset_name'] = np.zeros(batch_size)  # dummy, not used.
    new_gt['sample_index'] = sample_index
    # for debugging
    new_gt['vertices'] = new_vertices
    img, mask = uv_renderer(new_vertices.detach(),
                            g_cam.detach(), texture_pred)
    img = img.cpu()
    mask = mask.cpu()
    if backgrounds is not None:
        random_bg = backgrounds
        mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
        img = img * mask + random_bg * (1-mask)
    new_gt['img'] = torch.stack([normalize_T(img[i])
                                 for i in range(batch_size)])
    return new_gt


def compute_dt_barrier(mask, k=50):
    """
    Computes barrier distance transform of mask.
    """
    from scipy.ndimage import distance_transform_edt
    dist_out = distance_transform_edt(1-mask)
    dist_in = distance_transform_edt(mask)

    dist_diff = (dist_out - dist_in) / max(mask.shape)

    dist = 1. / (1 + np.exp(k * -dist_diff))
    return dist


def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad
