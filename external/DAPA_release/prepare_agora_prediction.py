"""
This script prepares the AGORA predictions in the following format:
    https://github.com/pixelite1201/agora_evaluation/blob/master/docs/prediction_format.md
"""


import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple, defaultdict
from tqdm import tqdm
import torchgeometry as tgm
import pickle

import config
import constants
from models import hmr
import smplx
import time
from datasets import BaseDataset
from utils.geometry import perspective_projection
# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--out_folder', default=None, help='Path to save the prediction pkl')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--split', required=True, choices=['validation', 'test'])

img_person_count = defaultdict(int)

def prepare_predictions(model, dataset, out_folder, batch_size=32, img_res=224, 
                        num_workers=32, shuffle=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to(device)
    t = time.time()
    print('Start initializing data loader')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) #, num_workers=num_workers)
    print('Initialized data loader in {} seconds'.format(time.time() - t))

    model_neutral = smplx.create(config.SMPL_MODEL_DIR, 
                                 model_type='smpl', gender='neutral', ext='npz').cuda()

    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        images = batch['img'].to(device)  # (B, 3, 224, 224)
        imgnames = batch['imgname']
        center = batch['center'].float()
        scale = batch['scale'].float() * 200
        curr_batch_size = images.shape[0]
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = model_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                                       global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_keypoints_3d_24 = pred_output.joints[:, :24]  # AGORA does matching using first 24 keypoints.
            
        # get the 2D joints
        camera_center = torch.zeros(curr_batch_size, 2)
        pred_cam_t = torch.stack([
            pred_camera[:,1], pred_camera[:,2],
            2*constants.FOCAL_LENGTH/(224 * pred_camera[:,0] +1e-9)
        ],dim=-1).cpu()
        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d_24.cpu(), translation=pred_cam_t,
            rotation=torch.eye(3).unsqueeze(0).expand(curr_batch_size, -1, -1),
            focal_length=constants.FOCAL_LENGTH, camera_center=camera_center)
        pred_keypoints_2d = pred_keypoints_2d / (224 / 2.)
        pred_keypoints_2d = 0.5 * 224 * (pred_keypoints_2d + 1)
        
        # move to original image pixel location
        pred_keypoints_2d = pred_keypoints_2d / 224. * scale.view(-1, 1, 1)
        pred_keypoints_2d[:, :, 0] = pred_keypoints_2d[:, :, 0] - scale.view(-1,1)/2 + center[:, 0].unsqueeze(1)
        pred_keypoints_2d[:, :, 1] = pred_keypoints_2d[:, :, 1] - scale.view(-1,1)/2 + center[:, 1].unsqueeze(1)
        
        for i in range(curr_batch_size):  # save one pkl file for each person
            output = {
                'pose2rot': False,  # everything is in matrix form
                'joints': pred_keypoints_2d[i].cpu().numpy(),
                'params': {
                    'transl': pred_cam_t[[i]].cpu().numpy(),  # (1, 3)
                    'betas': pred_betas[[i]].cpu().numpy(),  # (1, 10)
                    'global_orient': pred_rotmat[[i],0].unsqueeze(1).cpu().numpy(),  # (1,1,3,3)
                    'body_pose': pred_rotmat[[i],1:].cpu().numpy() # (1,23,3,3)
                }
            }
            imgname =  os.path.split(imgnames[i])[1]
            out_name = '{}_personId_{}.pkl'.format(imgname.strip(".png"), img_person_count[imgname])
            img_person_count[imgname] += 1
            out_name = os.path.join(out_folder, out_name)
            
            with open(out_name, 'wb') as f:
                pickle.dump(output, f)
    


if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    print('Loaded checkpoint at step count:', checkpoint['total_step_count'])
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    print('Loading dataset', 'agora_{}_keypoints'.format(args.split))
    dataset = BaseDataset(None, 'agora_{}_keypoints'.format(args.split), is_train=False)
    # Run evaluation
    prepare_predictions(model, dataset, args.out_folder,
                        batch_size=args.batch_size,
                        shuffle=args.shuffle)
