import os
import argparse
from collections import defaultdict
import glob
import json
import numpy as np
import scipy.io as sio
import tqdm
import config

def read_openpose(json_file):
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']
    keyp25 = np.reshape([people[i]['pose_keypoints_2d'] for i in range(len(people))], [-1, 25,3])
    return keyp25
    

def sample_frames(dataset_path):
    k_per_recording = 50  # 50 * 44 * 9 ~= 20000 images
    np.random.seed(42)
    months = range(6, 15)
    sampled_frames = defaultdict(dict)
    keypoints_files = defaultdict(dict)
    num_frames = 0
    for month in tqdm.tqdm(months):
        recordings = glob.glob(os.path.join(dataset_path, r'month{}/*_fps1'.format(month)))
        for recording_path in sorted(recordings):
            images = glob.glob(os.path.join(recording_path, 'images/*.jpg'))
            # starting from 98 with 3 frames interval, so there is no overlap between the training and test set.
            images = list(sorted(images))[98:-100:3] 

            if len(images) == 0: continue
            frame_names = np.random.choice(images, size=k_per_recording, replace=False)
            sampled_frames[month][recording_path] = frame_names  # full path to those frames

            op_paths = np.array([
                os.path.join(recording_path, 'keypoints/{}_keypoints.json'.format(os.path.split(frame_name)[-1].split('.')[0]))
                for frame_name in frame_names])
            
            keypoints_files[month][recording_path] = op_paths
            num_frames += k_per_recording

    print('Sampled {} frames in total.'.format(num_frames))
    return sampled_frames, keypoints_files
                                    
    

def seedlings_dataset_train_extract(frames_dict, keypoints_files, out_path):
    imgnames_, scales_, centers_, openposes_  = [], [], [], []
    scaleFactor = 1.2   # bbox expansion factor
    for month in tqdm.tqdm(frames_dict.keys()):
        for recording_path, frame_paths in frames_dict[month].items():
            keypoints_paths = keypoints_files[month][recording_path]
            for frame_path, kp_path in zip(frame_paths, keypoints_paths):
                try:
                    op_kps = read_openpose(kp_path)  # (num_ppl, 25, 3)
                except FileNotFoundError:
                    print(kp_path)
                    continue
                for p_i in range(op_kps.shape[0]):
                    kp_25 = op_kps[p_i]  # (25 * 3)
                    if kp_25[:, 2].sum() < 5.: continue # filter out low confidence detections
                    
                    # TODO: filter detection with imcomplete humans.
                    if len(kp_25[kp_25[:, 2]>0.5]) < 10: continue 
                    bbox = [min(kp_25[kp_25[:,2]>0,0]), min(kp_25[kp_25[:,2]>0,1]),
                            max(kp_25[kp_25[:,2]>0,0]), max(kp_25[kp_25[:,2]>0,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    
                    if abs(bbox[2]-bbox[0]) < 400 and abs(bbox[3]-bbox[1]) < 400:
                        continue
                    
                    # scaleFactor depends on the person height
                    torso_heights = []  # torso points: 2 (rshould), 9 (rhip), 5 (lshould), 12 (lhip)
                    if kp_25[9, 2]>0.5 and kp_25[2, 2]>0.5:
                        torso_heights.append(
                            np.linalg.norm(kp_25[9,:2] - kp_25[2,:2]))
                    if kp_25[5, 2]>0.5 and kp_25[12, 2]>0.5:
                        torso_heights.append(
                            np.linalg.norm(kp_25[5,:2] - kp_25[12,:2]))
                    
                    if len(torso_heights) == 0:
                        if kp_25[1,2]>0.5 and kp_25[8,2]>0.5:
                            torso_heights.append(
                                np.linalg.norm(kp_25[1,:2] - kp_25[8,:2]))
                        if len(torso_heights) == 0:
                            # Skip, person is too close.
                            continue
                
                    # Make torso 1/3 of the frame
                    scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * 1.2) / 200

                    imgnames_.append(frame_path)
#                     imgnames_.append(frame_path.replace('images', 'keypoints').replace('.jpg', '_rendered.png'))
                    centers_.append(center)
                    scales_.append(scale)
                    openposes_.append(kp_25)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'seedlings_dataset_train.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       openpose=openposes_)
    
    
if __name__ == '__main__':
    frames_dict, keypoints_files = sample_frames('data/seedlings')
    seedlings_dataset_train_extract(frames_dict, keypoints_files, config.DATASET_NPZ_PATH)
