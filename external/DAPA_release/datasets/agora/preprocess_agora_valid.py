import os
import argparse
from collections import defaultdict
import glob
import json
import numpy as np
import scipy.io as sio
import tqdm
import pickle5 as pickle
import config

def smpl_to_kp24(joints): # (45, 2)
    # map to the 24 gt kp used by SPIN
    part = np.zeros([24,3])
    part_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23])
    smpl_idx = np.array([8,5,2,1,4,7,21,19,17,16,18,20,24,26,25,28,27])
    part[part_idx, :2] = joints[smpl_idx]
    part[part_idx, 2] = 1
    return part


def agora_valid_extract(dataset_path, out_path):
    dataframe_path = os.path.join(dataset_path, 'validation_SMPL', 'SMPL')
    images_path = os.path.join(dataset_path, 'validation_images_3840x2160', 'validation') 

    # parts_ is the 2D keypoints. 
    imgnames_, scales_, centers_, parts_  = [], [], [], []
    poses_, shapes_ = [], []
    scaleFactor = 1.2   # bbox expansion factor
    valid_split = list(range(10))
    
    for split in tqdm.tqdm(valid_split):
        with open(os.path.join(dataframe_path, 'validation_{}_withjv.pkl'.format(split)), 'rb') as f:
            gt_data = pickle.load(f)
        for row_idx in range(gt_data.shape[0]):
            row = gt_data.iloc[row_idx]
            img_path = os.path.join(images_path, row.at['imgPath']) #.replace('.png', '_1280x720.png')
            isValid = row.at['isValid']
            valid_idxs = np.where(isValid)[0]
            num_persons = len(valid_idxs)  # TODO: try with different occlusion threshold
            
            for person_id in valid_idxs:
                smpl_path = os.path.join(dataset_path, row.at['gt_path_smpl'][person_id]).replace('.obj', '.pkl')
                gt = pickle.load(open(smpl_path, 'rb'))
                root_pose = gt['root_pose'].view(-1).detach().cpu().numpy()
                pose = gt['body_pose'].view(-1).detach().cpu().numpy()
                beta = gt['betas'].view(-1).detach().cpu().numpy()
                pose = np.concatenate([root_pose, pose])
                
                gt_joints_2d = row.at['gt_joints_2d'][person_id]  # (45, 2). No visibility flag.
                gt_joints_2d[:,0] = gt_joints_2d[:,0] # /3840*1280
                gt_joints_2d[:,1] = gt_joints_2d[:,1] # /2160*720
                part = smpl_to_kp24(gt_joints_2d)
                
                bbox = [min(part[part[:,2]>0,0]), min(part[part[:,2]>0,1]),
                        max(part[part[:,2]>0,0]), max(part[part[:,2]>0,1])]
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                
                # scaleFactor depends on the person height
                torso_heights = []  # torso points: 8 (rshould), 2 (rhip), 9 (lshould), 3 (lhip)
                torso_heights.append(np.linalg.norm(part[8] - part[2]))
                torso_heights.append(np.linalg.norm(part[9] - part[3]))
                # Make torso 1/3 of the frame
                scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
                
                imgnames_.append(img_path)
                scales_.append(scale)
                centers_.append(center)
                parts_.append(part)
                poses_.append(pose)
                shapes_.append(beta)
                
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'agora_valid.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       pose=poses_,
                       shape=shapes_,
            )

            
if __name__ == '__main__':
    dataset_path = 'data/agora'
    agora_valid_extract(dataset_path, config.DATASET_NPZ_PATH)
