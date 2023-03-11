"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```

Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error, compute_similarity_transform_batch
from utils.part_utils import PartRenderer

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p2', choices=[
    'seedlings', 'seedlings2', 'seedlings_visible', 'seedlings_inferred', 'seedlings_overall',
    'h36m-p1', 'h36m-p2', 'lsp', '3dpw', 
    'agora_valid', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=100, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    #renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))
    vertex_err = np.zeros(len(dataset))
    pa_vertex_err = np.zeros(len(dataset))
    mpjpe_scaled = np.zeros(len(dataset))  # seedlings: mpjpe / scale
    pck = np.zeros(len(dataset))  # seedlings: threshold is 1/5 the torso (1/3 of 224)
    pck_ka = np.zeros(len(dataset))
    error_per_joint = [np.zeros(len(dataset)) for _ in range(24)]

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name in ['seedlings', 'seedlings2', 'seedlings_visible', 'seedlings_inferred', 'seedlings_overall',
                        'h36m-p1', 'h36m-p2', '3dpw', 'agora_valid', 'mpi-inf-3dhp']:
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        if 'agora' in dataset_name: gt_pose[:,:3] = 0.
        
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]
        
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            if 'agora' in dataset_name: pred_rotmat[:,0] = torch.eye(3).cuda()
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            
        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_model = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas)
                gt_vertices = gt_model.vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 
                
                
                vertex_e = np.abs(gt_vertices.cpu().numpy() - pred_vertices.cpu().numpy()).sum(-1).mean(-1)
                vertex_err[step*batch_size:step*batch_size+curr_batch_size] = vertex_e
                
                pred_vertices_aligned = compute_similarity_transform_batch(pred_vertices.cpu().numpy(), gt_vertices.cpu().numpy())
                pa_vertex_e = np.abs(gt_vertices.cpu().numpy() - pred_vertices_aligned).sum(-1).mean(-1)
                pa_vertex_err[step*batch_size:step*batch_size+curr_batch_size] = pa_vertex_e


            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error
            
            
            if 'seedlings' in dataset_name:
                from utils.geometry import perspective_projection
                # for seedlings, only calculate 2d joint errors.
                gt_keypoints_2d = batch['keypoints'][:, 25:]  # (24, 3), last column is conf. score.
                pred_cam_t = torch.stack([
                        pred_camera[:,1], pred_camera[:,2],
                        2*constants.FOCAL_LENGTH/(224 * pred_camera[:,0] +1e-9)
                    ],dim=-1).cpu()
                camera_center = torch.zeros(curr_batch_size, 2)
                pred_keypoints_3d_24 = pred_output.joints[:, 25:]
                
                pred_keypoints_2d = perspective_projection(
                    pred_keypoints_3d_24.cpu(), translation=pred_cam_t,
                    rotation=torch.eye(3).unsqueeze(0).expand(curr_batch_size, -1, -1),
                    focal_length=constants.FOCAL_LENGTH, camera_center=camera_center)
                pred_keypoints_2d = pred_keypoints_2d / (224 / 2.)
                pred_keypoints_2d = 0.5 * 224 * (pred_keypoints_2d + 1)

                gt_keypoints_2d[:,:,:-1] = 0.5 * 224 * (gt_keypoints_2d[:, :, :-1] + 1)
                # pixel location in 224 * 224 image.
       
                error_2d_batch = np.sqrt(((gt_keypoints_2d[:,:,:2] - pred_keypoints_2d)**2).numpy().sum(axis=-1))  # 32, 24, 2
                for i_ in range(curr_batch_size):
                    error_2d = error_2d_batch[i_, gt_keypoints_2d[i_, :, 2].numpy()==1]
                    mpjpe[step * batch_size+i_] = error_2d.mean(axis=-1) # averaging over all joints.
                    error_2d_scaled =  error_2d / batch['scale'][i_].numpy()
                    mpjpe_scaled[step * batch_size+i_] = error_2d_scaled.mean(axis=-1) # averaging over all joints.
                    
                    thres = 224./3/5
                    num_annotated_kp = (gt_keypoints_2d[i_, :, 2]==1).sum()
                    if num_annotated_kp.item() > 0:
                        pck[step * batch_size + i_] = (error_2d < thres).sum() * 1. / num_annotated_kp.item()

                    for j_ in range(24):
                        if gt_keypoints_2d[i_, j_, 2]==1:
                            error_per_joint[j_][step * batch_size + i_] = error_2d_batch[i_, j_]
                        else:
                            error_per_joint[j_][step * batch_size + i_] = np.nan
                        

        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = renderer(pred_vertices, pred_camera)

        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                
                if '3dpw' in dataset_name or 'agora' in dataset_name:
                    print('Vertex Error: ' + str(1000 * vertex_err[:step * batch_size].mean()))
                    print('PA Vertex Error: ' + str(1000 * pa_vertex_err[:step * batch_size].mean()))
                print()
                    
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera, 
                 mpjpe=mpjpe, pck=pck, pck_ka=pck_ka, error_per_joint=error_per_joint)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        if 'seedlings' in dataset_name:
            print('MPJPE (2D in pixel): ' + str(mpjpe.mean()))
            print('MPJPE (scaled): ' + str(mpjpe_scaled.mean()))
            print('PCK (0.2*torso): ' + str(pck.mean()))
            print('PCK-knee-ankle: ' + str(pck_ka[~np.isnan(pck_ka)].mean()))
            print()
        else:
            print('MPJPE: ' + str(1000 * mpjpe.mean()))
            print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
            if '3dpw' in dataset_name:
                print('Vertex Error: ' + str(1000 * vertex_err.mean()))
                print('PA Vertex Error: ' + str(1000 * pa_vertex_err.mean()))
                print()
            
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()

if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    print('Loaded checkpoint at step count:', checkpoint['total_step_count'])
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
