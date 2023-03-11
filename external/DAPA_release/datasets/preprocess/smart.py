import os
import os.path as osp
import argparse
import numpy as np
import json
from tqdm import tqdm

SMART_DIR = '/scratch/users/zzweng/datasets/SMART/SportsCap'
IMG_DIR = os.path.join(SMART_DIR, 'images')
scaleFactor = 1.2  # bbox expansion factor

op_colors = [[255,     0,    85], # openpose color coding
    [255,     0,     0], 
    [255,    85,     0], 
    [255,   170,     0], 
    [255,   255,     0], 
    [170,   255,     0], 
    [85,   255,     0], 
    [0,   255,     0], 
    [255,     0,     0], 
    [0,   255,    85], 
    [0,   255,   170], 
    [0,   255,   255], 
    [0,   170,   255], 
    [0,    85,   255], 
    [0,     0,   255], 
    [255,     0,   170], 
    [170,     0,   255], 
    [255,     0,   255], 
    [85,     0,   255], 
    [0,     0,   255], 
    [0,     0,   255], 
    [0,     0,   255], 
    [0,   255,   255], 
    [0,   255,   255],
    [0,   255,   255]]


def xywh_to_center(xywh):
    x,y,w,h = xywh
    bbox  = [x,y,x+w,y+h]
    center = (bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2
    return center


def get_scale(joints, xywh):
    x,y,w,h = xywh
    bbox  = [x,y,x+w,y+h]
    torso_idxs = [[2,9],[5,12]]
    joints = np.array(joints)

    torso_heights = []  # torso points: 8 (rshould), 2 (rhip), 9 (lshould), 3 (lhip)
    for torso_idx in torso_idxs:
        # import pdb; pdb.set_trace()
        torso_heights.append(
            np.linalg.norm(joints[torso_idx[0],:2] - joints[torso_idx[1],:2]))

    scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
    return scale


def extract_dataset(out_path):
    # parts_ is the ground truth 2D keypoints, openpose_ is the openpose keypoints
    imgnames_, scales_, centers_, openpose_  = [], [], [], []
    anns_files = [
        # 'Table_VideoInfo_diving',  # this file is empty
        'Table_VideoInfo_gym',
        # 'Table_VideoInfo_polevalut_highjump_badminton'
        ]
    annotation = []
    for anns_file in anns_files:
        with open(os.path.join(SMART_DIR, 'annotations/{}.json'.format(anns_file)),'rb') as f:
            annotation.extend(json.load(f))
    print(len(annotation), 'video clips in total.')

    for idx, anns in tqdm(enumerate(annotation)):
        vid_len = len(anns['frames'])
        for i in range(vid_len):
            
            imgname = osp.join(IMG_DIR, anns['frames'][i]['img_name'])
            openpose = np.array(anns['frames'][i]['joints'])
            if openpose[:, 2].sum() < 10: continue

            center = xywh_to_center(anns['frames'][i]['bbox'])
            scale = get_scale(anns['frames'][i]['joints'], anns['frames'][i]['bbox'])

            imgnames_.append(imgname)
            openpose_.append(openpose)
            centers_.append(center)
            scales_.append(scale)
  
    if not os.path.isdir(out_path):
        os.makedirs(out_path)


    out_file = os.path.join(out_path, 'smart.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))

    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       openpose=openpose_)    
    print('Saved to', out_file)     
    
if __name__ == '__main__':
    extract_dataset('/home/groups/syyeung/zzweng/code/SPIN/data/dataset_extras')
