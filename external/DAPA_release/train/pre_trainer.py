from torchvision.utils import make_grid
import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import MixedDataset
from models import hmr, SMPL, get_texture_models
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat
from utils.renderer import Renderer
from utils.part_utils import PartRenderer
from utils import BaseTrainer
from train import Trainer
from train.train_utils import (set_grad,  uv_renderer, sample_textures, compute_dt_barrier)

import config
import constants
from .fits_dict import FitsDict

import time
from progress.bar import Bar
torch.manual_seed(0)  # TODO: seeting the seed. Move to options.
np.random.seed(0)


class PreTrainer(Trainer):

    def init_fn(self):
        self.train_ds = MixedDataset(
            self.options, ignore_3d=self.options.ignore_3d, is_train=True)
        print(self.train_ds.dataset_dict)
        self.model = hmr(config.SMPL_MEAN_PARAMS,
                         pretrained=True).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_shape_nr = nn.L1Loss(reduction='none').to(self.device)
        self.criterion_regr_nr = nn.MSELoss(reduction='none').to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        # Initialize SMPLify fitting module
        self.smplify = SMPLify(step_size=1e-2, batch_size=self.options.batch_size,
                               num_iters=self.options.num_smplify_iters, focal_length=self.focal_length)
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(
                checkpoint_file=self.options.pretrained_checkpoint)

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length,
                                 img_res=self.options.img_res, faces=self.smpl.faces)

        if self.options.pretrain and self.options.train_texture:
            texture_models = get_texture_models()
            self.model_texture = texture_models['model']
            self.optimizer_texture = texture_models['optimizer']
            self.texture_loss = texture_models['loss']
            self.texture_dt_loss_fn = texture_models['dt_loss']
            self.models_dict['model_texture'] = self.model_texture
            self.optimizers_dict['optimizer_texture'] = self.optimizer_texture
            self.tex_renderer = uv_renderer

    def finalize(self):
        self.fits_dict.save()

    def train_step(self, input_batch, real_flag=True):
        model = self.model
        optimizer = self.optimizer

        set_grad([model], True)
        model.train()

        # Get data from the batch
        images = input_batch['img']  # input image
        gt_keypoints_2d = input_batch['keypoints']  # 2D keypoints
        gt_pose = input_batch['pose']  # SMPL pose parameters
        gt_betas = input_batch['betas']  # SMPL beta parameters
        gt_joints = input_batch['pose_3d']  # 3D pose
        # flag that indicates whether SMPL parameters are valid
        has_smpl = input_batch['has_smpl'].byte()
        # flag that indicates whether 3D pose is valid
        has_pose_3d = input_batch['has_pose_3d'].byte()
        # flag that indicates whether image was flipped during data augmentation
        is_flipped = input_batch['is_flipped']
        # rotation angle used for data augmentation
        rot_angle = input_batch['rot_angle']
        # name of the dataset the image comes from
        dataset_name = input_batch['dataset_name']
        # index of example inside its dataset
        indices = input_batch['sample_index']
        batch_size = images.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(
            betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary
        if not real_flag:  # fake data has ground truth smpl parameters.
            opt_pose = gt_pose
            opt_betas = gt_betas
        else:
            opt_pose, opt_betas = self.fits_dict[(
                dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smpl(
            betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_pose[:, :3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * \
            self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(
            gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        opt_cam_t = estimate_translation(
            opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,
                                                       0.5 * self.options.img_res *
                                                       torch.ones(
                                                           batch_size, 2, device=self.device),
                                                       gt_keypoints_2d_orig).mean(dim=-1)

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera, feature = model(
            images, return_feature=True)

        pred_output = self.smpl(
            betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)

        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(
                                                       0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        if self.options.run_smplify and real_flag:
            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0, 0, 1], dtype=torch.float32,
                                                                                                    device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(
                pred_rotmat_hom).contiguous().view(batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices, new_opt_joints,\
                new_opt_pose, new_opt_betas,\
                new_opt_cam_t, new_opt_joint_loss = self.smplify(
                    pred_pose.detach(), pred_betas.detach(),
                    pred_cam_t.detach(),
                    0.5 * self.options.img_res *
                    torch.ones(batch_size, 2, device=self.device),
                    gt_keypoints_2d_orig)
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]

            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(
            ), is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())
        else:
            update = torch.zeros(batch_size, device=self.device).byte()

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters, if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(
            self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl

        opt_keypoints_2d = perspective_projection(opt_joints,
                                                  rotation=torch.eye(3, device=self.device).unsqueeze(
                                                      0).expand(batch_size, -1, -1),
                                                  translation=opt_cam_t,
                                                  focal_length=self.focal_length,
                                                  camera_center=camera_center)

        opt_keypoints_2d = opt_keypoints_2d / (self.options.img_res / 2.)

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(
            pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)
        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(
            pred_joints, gt_joints, has_pose_3d)
        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)
        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape +\
            self.options.keypoint_loss_weight * loss_keypoints +\
            self.options.keypoint_loss_weight * loss_keypoints_3d +\
            self.options.pose_loss_weight * loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas +\
            ((torch.exp(-pred_camera[:, 0]*10)) ** 2).mean()

        loss *= 60
        if self.options.train_texture and real_flag:
            texture_flow = self.model_texture.forward(feature.detach())
            images = images * \
                torch.tensor([0.229, 0.224, 0.225],
                             device=images.device).reshape(3, 1, 1)
            images = images + \
                torch.tensor([0.485, 0.456, 0.406],
                             device=images.device).reshape(3, 1, 1)
            texture_pred = sample_textures(texture_flow, images)

            opt_camera = torch.stack([
                2*self.focal_length /
                (opt_cam_t[:, 2]+1e-9)/self.options.img_res,
                opt_cam_t[:, 0], opt_cam_t[:, 1]], dim=-1)
            _, masks = self.tex_renderer(opt_vertices, opt_camera)

            tex_size = texture_pred.size(2)
            texture_pred = texture_pred.unsqueeze(
                4).repeat(1, 1, 1, 1, tex_size, 1)
            textured_images, masks_pred = self.tex_renderer(
                pred_vertices.detach(), pred_camera.detach(), texture=texture_pred)

            tex_loss = self.texture_loss(
                textured_images, images, masks_pred, masks)

            # Compute barrier distance transform.
            #mask_dts = np.stack([compute_dt_barrier(m) for m in masks.cpu().numpy()])
            #dt_tensor = torch.FloatTensor(mask_dts).cuda()
            # B x 1 x N x N
            #self.dts_barrier = Variable(dt_tensor, requires_grad=False).unsqueeze(1)
            #tex_dt_loss = self.texture_dt_loss_fn(texture_flow, self.dts_barrier)
            #tex_loss += tex_dt_loss

        if self.options.donot_train_on_fake and not real_flag:
            pass
        else:
            # Do backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'opt_vertices': opt_vertices,
                  'pred_cam_t': pred_cam_t.detach(),
                  'opt_cam_t': opt_cam_t,
                  'opt_camera': pred_camera.detach(),
                  'feature': feature.detach(),
                  'pred_rotmat': pred_rotmat.detach(),
                  'opt_pose': opt_pose.detach(),
                  'opt_betas': opt_betas.detach(),
                  'pred_keypoints_2d': pred_keypoints_2d.detach(),
                  'pred_joints': pred_joints.detach()
                  }
        output = dict(**input_batch, **output)
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()
                  }

        if self.options.train_texture and real_flag:
            self.optimizer_texture.zero_grad()
            tex_loss.backward()
            self.optimizer_texture.step()
            losses['tex_loss'] = tex_loss.detach().item()
            #losses['tex_dt_loss'] = tex_dt_loss.detach().item()
            output['texture_pred'] = texture_pred.detach()
            output['textured_images'] = textured_images.detach()

        return output, losses

    def fake_summaries(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * \
            torch.tensor([0.229, 0.224, 0.225],
                         device=images.device).reshape(1, 3, 1, 1)
        images = images + \
            torch.tensor([0.485, 0.456, 0.406],
                         device=images.device).reshape(1, 3, 1, 1)

        real_images = input_batch['real_img']
        real_images = real_images * \
            torch.tensor([0.229, 0.224, 0.225],
                         device=images.device).reshape(1, 3, 1, 1)
        real_images = real_images + \
            torch.tensor([0.485, 0.456, 0.406],
                         device=images.device).reshape(1, 3, 1, 1)

        pred_vertices = output['pred_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_vertices = output['opt_vertices']
        opt_cam_t = output['opt_cam_t']

        rendered_imgs = self.renderer.visualize_tb_with_synthetic(
            opt_vertices, opt_cam_t, pred_vertices, pred_cam_t, images, real_images)
        self.summary_writer.add_image(
            'synthetic/opt_shape', rendered_imgs, self.step_count)

        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(
                'synthetic/'+loss_name, val, self.step_count)
