import os, sys
import argparse
from collections import defaultdict, namedtuple
import glob
import json
import numpy as np
import scipy.io as sio
import tqdm
import pickle5 as pickle

import torch

import config



def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        keypoints.append(body_keypoints)

    return keypoints

def agora_valid_extract(dataset_path, out_path, split='validation'):
    images_path = os.path.join(dataset_path, split+'_images_3840x2160', split)
    keypoints_path = os.path.join(dataset_path, split+'_images_3840x2160', 'keypoints')
    
    # parts_ is the 2D keypoints. 
    imgnames_, scales_, centers_  = [], [], []
    scaleFactor = 1.2   # bbox expansion factor
    
    images = glob.glob(os.path.join(images_path, '*.png'))
    print(len(images), 'images')
    for img_path in tqdm.tqdm(images):
        img_name = os.path.split(img_path)[1].strip(".png")
        kp_path = os.path.join(keypoints_path, img_name+'_keypoints.json')
        parts = read_keypoints(kp_path)
        for person_idx in range(len(parts)):
            part = parts[person_idx]
            bbox = [min(part[part[:,2]>0,0]), min(part[part[:,2]>0,1]),
                            max(part[part[:,2]>0,0]), max(part[part[:,2]>0,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]

            # scaleFactor depends on the person height
            torso_heights = []  # torso points: 2 (rshould), 9 (rhip), 5 (lshould), 12 (lhip)
            if part[2,2] > 0 and part[9,2] > 0:
                torso_heights.append(np.linalg.norm(part[2,:2] - part[9,:2]))
            if part[5,2] > 0 and part[12,2] > 0:
                torso_heights.append(np.linalg.norm(part[5,:2] - part[12,:2]))
            # Make torso 1/3 of the frame
            if len(torso_heights) == 0:
                scale = max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor / 200
            else:
                scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
            
            if scale * 200 < 5: 
                import pdb; pdb.set_trace()
                continue
          
            imgnames_.append(img_path)
            scales_.append(scale)
            centers_.append(center)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'agora_{}_keypoints.npz'.format(split))
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_)
               
    
if __name__ == '__main__':
    dataset_path = 'data/agora'
    split = sys.argv[1]
    print('Extracting {} split...'.format(split))
    agora_valid_extract(dataset_path, config.DATASET_NPZ_PATH, split)
