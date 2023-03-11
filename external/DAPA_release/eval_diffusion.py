"""
Example usage:
```
python eval_diffusion.py --dataset=pole_vault --eval_type=3d --checkpoint=data/model_checkpoint.pt 

```

"""
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from utils.geometry import perspective_projection
from utils.pose_utils import reconstruction_error, compute_similarity_transform_batch

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to network checkpoint')
parser.add_argument('--dataset', default='pole_vault', help='Choose evaluation dataset')
parser.add_argument('--eval_type', default='3d', choices=["2d", "3d_keypoints", "3d_smpl", "2d_25"])
parser.add_argument('--log_freq', default=100, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

def run_evaluation(model, dataset, result_file,
                   batch_size=32, img_res=224, eval_type='3d',
                   num_workers=32, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    vertex_err = np.zeros(len(dataset))
    pa_vertex_err = np.zeros(len(dataset))
    pck = np.zeros(len(dataset))
    error_per_joint = [np.zeros(len(dataset)) for _ in range(24)]

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    joint_mapper_h36m = constants.H36M_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = images.shape[0]
        
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(images)
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()

        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)

        if eval_type == '3d_smpl' or eval_type == '3d_keypoints':
            if eval_type == '3d_smpl':
                gt_model = smpl_neutral(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas)
                gt_vertices = gt_model.vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
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
                pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
                pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            elif eval_type == '3d_keypoints':
                gt_keypoints_3d = batch['pose_3d'].to(device)
                gt_keypoints_3d, has_3d_keypoint = gt_keypoints_3d[:, :, :3], gt_keypoints_3d[:, :, 3]
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

                pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
                if save_results:
                    pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :] = pred_keypoints_3d.cpu().numpy()
                pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            pred_keypoints_3d = pred_keypoints_3d[:, [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16], :]
            gt_keypoints_3d = gt_keypoints_3d[:, [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16], :]
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error
            
        elif eval_type == "2d" or eval_type == "2d_25":
            pred_cam_t = torch.stack([
                    pred_camera[:,1], pred_camera[:,2],
                    2*constants.FOCAL_LENGTH/(224 * pred_camera[:,0] +1e-9)
                ],dim=-1).cpu()
            camera_center = torch.zeros(curr_batch_size, 2)
            # only calculate 2d joint errors.
            if eval_type == "2d_25":
                gt_keypoints_2d = batch['keypoints'][:, :25]
                pred_keypoints_3d = pred_output.joints[:, :25]
            else:
                gt_keypoints_2d = batch['keypoints'][:, 25:]  # (24, 3), last column is conf. score.
                pred_keypoints_3d = pred_output.joints[:, 25:]
            
            pred_keypoints_2d = perspective_projection(
                pred_keypoints_3d.cpu(), translation=pred_cam_t,
                rotation=torch.eye(3).unsqueeze(0).expand(curr_batch_size, -1, -1),
                focal_length=constants.FOCAL_LENGTH, camera_center=camera_center)
            pred_keypoints_2d = pred_keypoints_2d / (224 / 2.)
            pred_keypoints_2d = 0.5 * 224 * (pred_keypoints_2d + 1)
            gt_keypoints_2d[:,:,:-1] = 0.5 * 224 * (gt_keypoints_2d[:, :, :-1] + 1)

            error_2d_batch = np.sqrt(((gt_keypoints_2d[:,:,:2] - pred_keypoints_2d)**2).numpy().sum(axis=-1))  # 32, 24, 2
            for i_ in range(curr_batch_size):
                error_2d = error_2d_batch[i_, gt_keypoints_2d[i_, :, 2].numpy()==1]
                mpjpe[step * batch_size+i_] = error_2d.mean(axis=-1) # averaging over all joints.

                thres = 224./10
                num_annotated_kp = (gt_keypoints_2d[i_, :, 2]==1).sum()
                if num_annotated_kp.item() > 0:
                    pck[step * batch_size + i_] = (error_2d < thres).sum() * 1. / num_annotated_kp.item()

                for j_ in range(24):
                    if gt_keypoints_2d[i_, j_, 2]==1:
                        error_per_joint[j_][step * batch_size + i_] = error_2d_batch[i_, j_]
                    else:
                        error_per_joint[j_][step * batch_size + i_] = np.nan
                    
        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_type == "3d":
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print('Vertex Error: ' + str(1000 * vertex_err[:step * batch_size].mean()))
                print('PA Vertex Error: ' + str(1000 * pa_vertex_err[:step * batch_size].mean()))

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera, 
                 mpjpe=mpjpe, pck=pck, error_per_joint=error_per_joint)
    # Print final results during evaluation
    print('*** Final Results ***')
    if eval_type == "2d" or eval_type == "2d_25":
        print('PCK: ' + str(pck.mean() * 100))
        print()
        
        is_pole_vault = False
        if is_pole_vault: # print pck per video
            vid_names = [os.path.split(imgname)[1].split('-')[0] for imgname in dataset.imgname]
            vids = sorted(set(vid_names))
            for vid_name in vids:
                vid_idx = [i for i, x in enumerate(vid_names) if x == vid_name]
                print(vid_name + ': ' + str(pck[vid_idx].mean()))
        return pck.mean()
    else:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print()
        return recon_err.mean() * 1000


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
    run_evaluation(model, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle, eval_type=args.eval_type,
                   log_freq=args.log_freq)
