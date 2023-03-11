from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
import torch
import torch.nn as nn
from torchgeometry import rotation_matrix_to_angle_axis
import wandb
from datetime import date

import config
import constants
from .fits_dict import FitsDict
from datasets import MixedDataset
from models import hmr, SMPL, get_texture_models
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat
from utils.renderer import Renderer
from train import Trainer
from train.train_utils import (set_grad, synthetic_gt, uv_renderer, sample_textures)

class AdaptTrainer(Trainer):
    def init_fn(self):
        self.train_ds = MixedDataset(
            self.options, ignore_3d=self.options.ignore_3d, is_train=True, parse_random_bg=True)
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
        self.summary_writer = wandb.init(
            project=self.options.wandb_project, name=self.options.name, 
            config=vars(self.options), group=date.today().strftime("%B %d"))

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

        if self.options.vposer:
            expr_dir = config.VPOSER_PATH
            vp, _ = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
            vp = vp.to('cuda')
            self.vp = vp

        if self.options.use_texture:
            texture_models = get_texture_models()
            self.model_texture = texture_models['model']
            self.optimizer_texture = texture_models['optimizer']
            self.texture_loss = texture_models['loss']
            self.models_dict['model_texture'] = self.model_texture
            self.optimizers_dict['optimizer_texture'] = self.optimizer_texture
            self.tex_renderer = uv_renderer

    def finalize(self):
        self.fits_dict.save()

    def _generate_fake_from_rotmat(self, real_batch, g_rotmat):
        batch_size = g_rotmat.size(0)
        g_rotmat_hom = torch.cat(
            [g_rotmat.view(-1, 3, 3),
             torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).view(
                 1, 3, 1).expand(batch_size*24, -1, -1)
             ], dim=-1)
        g_pose = rotation_matrix_to_angle_axis(
            g_rotmat_hom).contiguous().view(batch_size, -1)
        g_pose[torch.isnan(g_pose)] = 0.0

        texture_pred = None if 'texture_pred' not in real_batch else real_batch[
            'texture_pred']
        backgrounds = None if not self.options.add_background else real_batch['random_bg'].cpu(
        )

        fake_data = synthetic_gt(g_rotmat, g_pose, real_batch['opt_betas'],
                                 real_batch['opt_camera'], real_batch['pred_rotmat'],
                                 self.smpl, self.options, use_opt_global_orient=self.options.use_opt_global_orient,
                                 texture_pred=texture_pred,
                                 backgrounds=backgrounds, sample_index=real_batch['sample_index']
                                 )
        fake_data['real_img'] = real_batch['img']
        return fake_data

    def get_fake(self, real_batch):
        vp = self.vp
        # If using noised feature for data augmentation, then do this
        if not self.options.run_smplify:
            input_pose = real_batch['pred_pose'].to(self.device)
        else:
            input_pose = real_batch['opt_pose'].to(self.device)
        vposer_input = input_pose[:, 3:66]
        N = input_pose.size(0)
        
        # Recon with perturbation
        q_z = vp.encode(vposer_input)
        q_z_sample = q_z.rsample()
        if self.options.g_input_noise_type == 'randn':
            q_z_sample += self.options.g_input_noise_scale * torch.randn_like(q_z_sample)
            decode_results = vp.decode(q_z_sample)
            vposer_output = decode_results['pose_body'].contiguous().view(N, -1)
            input_pose[:, 3:66] = vposer_output
            
        elif self.options.g_input_noise_type == 'mul':
            q_z_sample *= 1 + (self.options.g_input_noise_scale * torch.rand_like(q_z_sample))
            decode_results = vp.decode(q_z_sample)
            vposer_output = decode_results['pose_body'].contiguous().view(N, -1)
            input_pose[:, 3:66] = vposer_output
            
        elif self.options.g_input_noise_type == 'fgsm': 
            q_z_0 = q_z_sample.detach().clone().requires_grad_()
            decode_results = vp.decode(q_z_0)
            vposer_output = decode_results['pose_body'].contiguous().view(N, -1)
            input_pose[:, 3:66] = vposer_output
            g_rotmat0 = batch_rodrigues(input_pose.view(-1, 3)).view(-1, 24, 3, 3)
            fake_data = self._generate_fake_from_rotmat(
                real_batch, g_rotmat0)
            fake_loss0 = self.get_2d_loss(fake_data, False, True)
            fake_loss0.backward()
            q_z_0_grad = q_z_0.grad
            adv_q_z = q_z_0 + self.options.g_input_noise_scale * \
                torch.sign(q_z_0_grad).float()
            q_z_sample = adv_q_z
            decode_results = vp.decode(q_z_sample)
            vposer_output = decode_results['pose_body'].contiguous().view(N, -1)
            input_pose[:, 3:66] = vposer_output
            
        elif self.options.g_input_noise_type == 'rand_sample':
            vposer_output = vp.sample_poses(num_poses=N)['pose_body'].contiguous().view(N, -1)
            
        else:
            raise ValueError()
            
#         vposer_samples = vp.sample_poses(num_poses=N)['pose_body'].contiguous().view(N, -1)
#         input_pose[:, 3:66] = vposer_output
#         input_pose[:, 60:66] = vposer_samples[:, 60-3:66-3]
#         input_pose[:, 21:27] = vposer_samples[:, 21-3:27-3]
        g_rotmat = batch_rodrigues(input_pose.view(-1, 3)).view(-1, 24, 3, 3)
        fake_data = self._generate_fake_from_rotmat(real_batch, g_rotmat)

        return fake_data

    def get_2d_loss(self, input_batch, real_flag, reduce):
        set_grad([self.model], False)
        self.model.eval()
        input_batch = {k: v.to('cuda') if isinstance(
            v, torch.Tensor) else v for k, v in input_batch.items()}
        # Get data from the batch
        images = input_batch['img']  # input image
        gt_keypoints_2d = input_batch['keypoints']
        batch_size = images.shape[0]

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)
        pred_output = self.smpl(
            betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices.detach()
        pred_joints = pred_output.joints.detach()
        # pred_camera = pred_camera.detach()
        pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)
        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(
                                                       0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length, camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        # Compute loss
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight, reduce=reduce)  # , reduce=real_flag)

        # If real data, calculate loss using opt params (from the SPIN loop); If fake, use gt params.
        if real_flag:
            opt_output = self.smpl(betas=input_batch['opt_betas'],
                                   body_pose=input_batch['opt_pose'][:,
                                                                     3:], global_orient=input_batch['opt_pose'][:, :3],
                                   reduce=reduce)
            opt_vertices = opt_output.vertices
            opt_joints = opt_output.joints
            opt_joints = torch.cat([opt_joints,
                                    torch.ones((batch_size, opt_joints.shape[1], 1)).to(self.device)], 2)
            ones_flag = torch.ones(batch_size, device=self.device).byte()
            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas,
                                                               input_batch['opt_pose'], input_batch['opt_betas'],
                                                               ones_flag, reduce=reduce)
            loss_keypoints_3d = self.keypoint_3d_loss(
                pred_joints, opt_joints[:, :25], ones_flag, reduce=reduce)
            # Per-vertex loss for the shape
            loss_shape = self.shape_loss(
                pred_vertices, opt_vertices, ones_flag, reduce=reduce)
        else:
            gt_out = self.smpl(betas=input_batch['betas'],
                               body_pose=input_batch['pose'][:,
                                                             3:], global_orient=input_batch['pose'][:, :3],
                               reduce=reduce)
            # for fake data, gt_model_joints = input_batch['pose_3d']
            gt_model_joints = gt_out.joints
            gt_vertices = gt_out.vertices
            ones_flag = torch.ones(batch_size, device=self.device).byte()
            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas,
                                                               input_batch['pose'], input_batch['betas'],
                                                               ones_flag, reduce=reduce)
            loss_keypoints_3d = self.keypoint_3d_loss(
                pred_joints, input_batch['pose_3d'], ones_flag, reduce=reduce)
            # Per-vertex loss for the shape
            loss_shape = self.shape_loss(
                pred_vertices, gt_vertices, ones_flag, reduce=reduce)

        # Note: this is used to update the Generator!!! So the loss for real sample should contain no gradient information
        #
        # Compute total loss
        loss = self.options.shape_loss_weight * loss_shape +\
            self.options.keypoint_loss_weight * loss_keypoints +\
            self.options.keypoint_loss_weight * loss_keypoints_3d +\
            self.options.pose_loss_weight * loss_regr_pose + 0 * loss_regr_betas

        loss *= 60
        set_grad([self.model], True)
        self.model.train()
        return loss

    def keypoint_3d_loss_synthetic(self, pred_keypoints_3d, gt_keypoints_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, :25, :]
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] +
                         gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (
                pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses_synthetic(self, pred_rotmat, gt_pose):
        gt_rotmat = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss_regr_pose = self.criterion_regr(pred_rotmat, gt_rotmat)
        return loss_regr_pose

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
        opt_global_orient = opt_pose[:, :3]
        opt_output = self.smpl(
            betas=opt_betas, body_pose=opt_pose[:, 3:], global_orient=opt_global_orient)
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints
        # """

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * \
            self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera, feature = model(
            images, return_feature=True)

        pred_global_orient = pred_rotmat[:, 0].unsqueeze(1)
        pred_output = self.smpl(
            betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_global_orient, pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)

        camera_center = torch.zeros(batch_size, 2, device=self.device)

        # if self.adapt_mode or not real_flag:
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
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(
                                                       0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        # Convert predicted rotation matrices to axis-angle
        pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor(
            [0, 0, 1], dtype=torch.float32, device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
        pred_pose = rotation_matrix_to_angle_axis(
            pred_rotmat_hom).contiguous().view(batch_size, -1)
        # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
        pred_pose[torch.isnan(pred_pose)] = 0.0

        # for SPIN-ft, do not run smplify.
        if self.options.run_smplify and real_flag: # and not (self.options.agora and not self.options.ignore_3d):
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
#             update = torch.ones(batch_size, device=self.device).byte()

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

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)
        # Compute 3D keypoint loss
        if real_flag and not self.options.run_smplify:
            # only use 2D keypoints on the real target dataset.
            loss_keypoints_3d = torch.tensor(0.).cuda()
            loss_shape = loss_regr_pose = loss_regr_betas = torch.tensor(
                0.).cuda()
        elif real_flag and self.options.run_smplify:
            opt_joints = torch.cat([opt_joints,
                                    torch.ones((batch_size, opt_joints.shape[1], 1)).to(self.device)], 2)
            loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, opt_joints[:, :25],
                                                      torch.ones(batch_size, device=self.device).byte())
            loss_shape = self.shape_loss(
                pred_vertices, opt_vertices, valid_fit)
            loss_regr_pose, loss_regr_betas = self.smpl_losses(
                pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)
        else:  # for fake data, we have the gt
            loss_keypoints_3d = self.keypoint_3d_loss(
                pred_joints, gt_joints, has_pose_3d)
            loss_shape = self.shape_loss(
                pred_vertices, opt_vertices, valid_fit)
            loss_regr_pose, loss_regr_betas = self.smpl_losses(
                pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape +\
            self.options.keypoint_loss_weight * loss_keypoints +\
            self.options.keypoint_loss_weight * loss_keypoints_3d +\
            self.options.pose_loss_weight * loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas +\
            ((torch.exp(-pred_camera[:, 0]*10)) ** 2).mean()
        loss *= 60

        # undo normalization
        images = images * \
            torch.tensor([0.229, 0.224, 0.225],
                         device=images.device).reshape(3, 1, 1)
        images = images + \
            torch.tensor([0.485, 0.456, 0.406],
                         device=images.device).reshape(3, 1, 1)

        if self.options.train_texture and real_flag:
            texture_flow = self.model_texture.forward(feature.detach())
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
        output['pred_pose'] = pred_pose.detach()
        output = dict(**input_batch, **output)

        if self.options.use_texture and real_flag:
            # use the pretrained texture model to predict the texture from the real images.
            with torch.no_grad():
                texture_flow = self.model_texture.forward(feature.detach())
                texture_pred = sample_textures(texture_flow, images)
                tex_size = texture_pred.size(2)
                texture_pred = texture_pred.unsqueeze(
                    4).repeat(1, 1, 1, 1, tex_size, 1)
                output['texture_pred'] = texture_pred

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
        self.summary_writer.log({'synthetic/opt_shape': wandb.Image(rendered_imgs)})

        for loss_name, val in losses.items():
            self.summary_writer.log({'synthetic/'+loss_name: val})