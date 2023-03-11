"""API for essential utilities of HMR.
"""

import cv2
import os
import pickle
from tqdm import tqdm
from loguru import logger

from PIL import Image
import numpy as np
import torch
import torchgeometry as tgm
from torchvision.transforms import Normalize, ToPILImage, ToTensor, Resize, Compose


import config, constants
from models import hmr, SMPL
from utils.geometry import perspective_projection
from utils.renderer import Renderer
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

expr_dir = '../../data/V02_05'

smpl_model = SMPL(
            os.path.join(os.path.dirname(__file__), config.SMPL_MODEL_DIR), batch_size=1, create_transl=False
        )


class SPIN_wrapper(object):
    def __init__(self, checkpoint_path, device='cuda') -> None:
        smpl_model.to(device)
        renderer = Renderer(faces = smpl_model.faces)
        img_transform = Compose([
            ToTensor(),
            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD),
            Resize((224, 224)),
        ])
        vp, _ = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
        vp = vp.to('cuda')

        model = hmr(config.SMPL_MEAN_PARAMS)
        checkpoint = torch.load(checkpoint_path)
        print('Loaded checkpoint at step count:', checkpoint['total_step_count'])
        model.load_state_dict(checkpoint['model'], strict=False)
        model.to(device)

        self.model = model
        self.renderer = renderer
        self.img_transform = img_transform
        self.smpl_model = smpl_model
        self.vp = vp
        self.device = device
        self.focal_length = 5000
        self.img_res = 224

    def inference_single(self, img, mask=None, move_person_to_center=False):
        if move_person_to_center:
            assert mask is not None, 'Mask is required when move_person_to_center is True'
            pos_idxs = np.where(np.array(mask) > 0)
            x_min = pos_idxs[1].min()
            x_max = pos_idxs[1].max()
            y_min = pos_idxs[0].min()
            y_max = pos_idxs[0].max()
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            size = max(x_max - x_min, y_max - y_min) * 1.2
            img = img.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))
            
        input_tensor = self.img_transform(img)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = self.model(input_tensor)

        # if move_person_to_center:
            # pred_camera = self.sample_center_camera()

        rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=self.device).view(1,3,1)
        rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(1 * 24, -1, -1)), dim=-1)
        pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)

        _, depth = self.render(None, pred_pose, pred_betas, pred_camera, input_tensor, None, None, None, None)
        return depth

    def inference(
        self, args, in_folder_name, out_folder_name, apply_pose_aug=True, num_aug_samples=1, save_grid=False,
        move_person_to_center=False):
        logger.info(f'Running SPIN inference on {in_folder_name} and save to {out_folder_name}')
        self.model.eval()

        files = os.listdir(os.path.join(args.out_path, in_folder_name))
        files = sorted(files)
        annotations = {}
        for file in tqdm(files):
            img_path = os.path.join(args.out_path, in_folder_name, file)
            if img_path.endswith('_grid.png'): continue
            img = Image.open(img_path).convert('RGB')

            if move_person_to_center:
                mask_path = img_path.replace('init_images', 'masks')
                if not os.path.exists(mask_path):
                    continue
                mask = Image.open(mask_path).convert('L')
                pos_idxs = np.where(np.array(mask) > 0)
                x_min = pos_idxs[1].min()
                x_max = pos_idxs[1].max()
                y_min = pos_idxs[0].min()
                y_max = pos_idxs[0].max()
                center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                size = max(x_max - x_min, y_max - y_min) * 1.2
                img = img.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))
                
            input_tensor = self.img_transform(img)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = self.model(input_tensor)

            # if move_person_to_center:
                # pred_camera = self.sample_center_camera()

            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=self.device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(1 * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)

            status = self.render(args, pred_pose, pred_betas, pred_camera, input_tensor, file, out_folder_name, sample_idx=0, save_grid=save_grid)
            out_file_name = f'{os.path.splitext(file)[0]}_0.png'
            if status > 0:
                annotations[out_file_name] = self.parse_gt(pred_pose, pred_betas, pred_camera)
            else:
                print(f'Failed to render {file}')
            
            if apply_pose_aug:
                init_pose = pred_pose.clone()
                init_betas = pred_betas.clone()
                for sample_idx in range(1, num_aug_samples):
                    pred_pose, pred_betas = self.apply_augmentation(init_pose.clone(), init_betas.clone())
                    status = self.render(
                        args, pred_pose, pred_betas, pred_camera, input_tensor, file, 
                        out_folder_name, sample_idx, save_grid=save_grid)
                    out_file_name = f'{os.path.splitext(file)[0]}_{sample_idx}.png'
                    if status > 0:
                        annotations[out_file_name] = self.parse_gt(pred_pose, pred_betas, pred_camera)
                    else:
                        print(f'Failed to render {file}')
        return annotations

    def parse_gt(self, pred_pose, pred_betas, pred_camera):
        """ Returns the 2D keypoints.
        """
        batch_size = 1
        smpl_out = self.smpl_model(betas=pred_betas, body_pose=pred_pose[:,3:], global_orient=pred_pose[:,:3])
        pred_joints = smpl_out.joints

        pred_cam_t = torch.stack([pred_camera[:, 1], pred_camera[:, 2],
                                  2*self.focal_length/(self.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)
        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pj2d = perspective_projection(pred_joints,
                                      rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                      translation=pred_cam_t,
                                      focal_length=self.focal_length, camera_center=camera_center)

        return {'pj2d_org': pj2d.cpu().numpy(), 'thetas': pred_pose.cpu().numpy(), 'betas': pred_betas.cpu().numpy(), 'cam': pred_camera.cpu().numpy()}

    def apply_augmentation(self, pred_pose, pred_betas):
        thetas_z = self.vp.encode(pred_pose[:, 3:-6]).rsample()
        # thetas_z *= 1.5
        thetas_z *= (1.5 + 0.3 * torch.randn_like(thetas_z, device=self.device))
        aug_thetas = self.vp.decode(thetas_z)
        pred_pose[:, 3:-6] = aug_thetas['pose_body'].reshape(-1, 63)

        # Augment orientation.
        # pred_pose[:, :3] = pred_pose[:, :3] * (1 + 0.5 * torch.randn_like(pred_pose[:, :3], device=self.device))
        pred_pose[:, :3] += torch.randn(3).to(self.device) * np.radians(10)

        pred_betas = pred_betas * (1 + 0.1 * torch.randn_like(pred_betas, device=self.device))
        return pred_pose, pred_betas

    def render(self, args, pred_pose, pred_betas, pred_camera, input_tensor, file, out_folder_name, sample_idx,
                save_grid=False):
        smpl_out = self.smpl_model(betas=pred_betas, body_pose=pred_pose[:,3:], global_orient=pred_pose[:,:3])

        new_vertices = smpl_out.vertices
        images = input_tensor * torch.tensor(constants.IMG_NORM_STD, device=self.device).reshape(1,3,1,1)
        images = images + torch.tensor(constants.IMG_NORM_MEAN, device=self.device).reshape(1,3,1,1)
        cam = torch.stack([pred_camera[:,1], pred_camera[:,2], 
                            2*self.focal_length/(self.img_res * pred_camera[:,0] +1e-9)],dim=-1).cpu()
    
        depth = self.renderer.visualize_depth(new_vertices.cuda(), cam.cuda())

        # scale the depth values so the span in z axis is 60
        idx_to_modify = (depth > 0)
        if idx_to_modify.sum() > 0:
            depth_range = depth.max() - depth[idx_to_modify].min()
            new = ((depth[idx_to_modify] - depth[idx_to_modify].min()) * 60 / depth_range)
            depth[idx_to_modify] = new + 150
        else:
            return -1  # it means that the model is not visible in the image

        img = self.renderer.visualize_tb(new_vertices.cuda(), cam.cuda(), images)
        img = img.cpu()

        if args is None:
            return img, depth
        else:
            if save_grid:
                ToPILImage()(img).save(os.path.join(args.out_path, out_folder_name, f'{os.path.splitext(file)[0]}_grid.png'))

            cv2.imwrite(os.path.join(args.out_path, out_folder_name, f'{os.path.splitext(file)[0]}_{sample_idx}.png'), depth.cpu().numpy())
            return 1

    def sample_center_camera(self):
        device = self.device
        dtype = torch.float32
        camera = torch.zeros(1, 3).to(device=device, dtype=dtype)
        # sample a random uniform value in the range of [0.2, 0.3]
        # camera[:, 0] = np.random.rand() * 0.1 + 0.2  # smaller value means further away from the camera.
        camera[:, 0] = 0.3
        camera[:, 1] = 0.  # - means move to the left. + means move to the right.
        camera[:, 2] = 0.2  # move the person slight down. 
        return camera

    def finetune(self, annot_name, generation_path, image_folder, num_epochs, batch_size=32, lr=1e-5):
        self.model.train()

        dataset = DatasetFromGen(annot_name, generation_path, image_folder, self.img_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        def loss_function(output, target):
            return torch.mean((output - target) ** 2)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_acc = LossAccumulator()
        for epoch in range(num_epochs):
            loss_acc.reset()
            for _, batch in enumerate(dataloader):
                image, thetas, betas, cam = batch['image'], batch['thetas'], batch['betas'], batch['cam']
                cur_batch_size = image.shape[0]
                image, thetas, betas, cam = image.to(self.device), thetas.to(self.device), betas.to(self.device), cam.to(self.device)
                optimizer.zero_grad()

                pred_rotmat, pred_betas, pred_camera = self.model(image)
                
                rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=self.device).view(1,3,1)
                rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(cur_batch_size*24, -1, -1)), dim=-1)
                pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
                
                loss = loss_function(pred_betas, betas) + loss_function(pred_pose, thetas) + loss_function(pred_camera, cam)
                loss.backward()
                optimizer.step()
                loss_acc.add(loss.item())

            logger.info(f'Epoch: {epoch}. Loss stats: {loss_acc}.')
        

class DatasetFromGen(torch.utils.data.Dataset):
    def __init__(self, annot_name, gen_path, image_folder, transform):
        self.gen_path = gen_path
        image_path = os.path.join(gen_path, image_folder)
        image_files = os.listdir(image_path)
        image_files = [file for file in image_files if not file.endswith('_grid.png')]
        with open(os.path.join(gen_path, annot_name), 'rb') as f:
            smpl_gt = pickle.load(f)

        self.image_files = image_files
        self.smpl_gt = smpl_gt
        self.transform = transform
        self.image_folder = image_folder

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = cv2.imread(os.path.join(self.gen_path, self.image_folder, image_file), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        pose = self.smpl_gt[image_file]['thetas']
        betas = self.smpl_gt[image_file]['betas']
        pj2d_org = self.smpl_gt[image_file]['pj2d_org']
        camera = self.smpl_gt[image_file]['cam']
        return {'image': image, 'thetas': pose, 'betas': betas, 'cam': camera, 'pj2d_org': pj2d_org}


# loss accumulator
class LossAccumulator():
    def __init__(self):
        self.losses = []
        self.losses_avg = []
        self.losses_std = []
        self.losses_min = []
        self.losses_max = []

    def __str__(self):
        self.compute()
        return f'avg: {self.losses_avg[-1]:.4f}, std: {self.losses_std[-1]:.4f}'

    def add(self, loss):
        self.losses.append(loss)

    def compute(self):
        self.losses_avg.append(np.mean(self.losses))
        self.losses_std.append(np.std(self.losses))
        self.losses_min.append(np.min(self.losses))
        self.losses_max.append(np.max(self.losses))
        self.losses = []

    def get_avg(self):
        return self.losses_avg

    def get_std(self):
        return self.losses_std

    def get_min(self):
        return self.losses_min

    def get_max(self):
        return self.losses_max

    def get_last(self):
        return self.losses[-1]

    def get_all(self):
        return self.losses

    def reset(self):
        self.losses = []
        self.losses_avg = []
        self.losses_std = []
        self.losses_min = []
        self.losses_max = []
