"""Create dataset in format accepted by SPIN/DAPA.
"""

import os
import json
import numpy as np
import tqdm

base_path = r'./data/SportsCap_Dataset_SMART_v1'
files = [
    'Table_VideoInfo_diving.json', 
    'Table_VideoInfo_polevalut_highjump_badminton.json', 
    'Table_VideoInfo_gym.json']

scaleFactor = 1.2   # bbox expansion factor

def create_gymnastics_dataset(out_path, action="balancebeam", split="train"):
    if action == 'balancebeam':
        action_label = '184;204'  # 20 videos, 444 frames. 
        train_len = 5
    elif action == 'vault':
        action_label = 'vt'  # 11 videos, 242 frames.
        train_len = 4
    elif action == 'unevenbars':
        action_label = '271'
        train_len = 3
    else:
        # TODO: add other actions
        raise NotImplementedError()

    file = files[2]
    annot_path = os.path.join(base_path, 'annotations', file)
    with open(annot_path) as f:
        data = json.load(f)
    imgnames_, scales_, centers_, part_  = [], [], [], []
    
    valid_actions = [vid['action_labels'] for vid in data]
    print(set(valid_actions))
    valid_videos = [vid['VideoName'] for vid in data if vid['action_labels'] in action_label]
    
    if split == "train":
        valid_labels = valid_videos[:train_len]
        print(f"Total {len(valid_videos)} videos. Using {len(valid_labels)} for training.")
    else:
        valid_labels = valid_videos[train_len:]
        print(f"Total {len(valid_videos)} videos. Using {len(valid_labels)} for testing.")

    for vid in data:
        if vid['VideoName'] not in valid_labels: continue

        for meta in vid['frames']:
            if len(meta['joints']) == 0: continue
            img_path = os.path.join(base_path, 'images', meta['img_name'])
            openpose = np.array(meta['joints'])
            openpose[:, 2] = 1 - openpose[:, 2]  # 0 is visible.
            openpose[15:, 2] = 0.

            part = np.zeros([24, 3])
            part_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23])
            part[part_idx, :] = openpose[[11,10,9,12,13,14,4,3,2,5,6,7,0,16,15,18,17]]

            bbox = [min(openpose[:,0]), min(openpose[:,1]),
                    max(openpose[:,0]), max(openpose[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            
            # scaleFactor depends on the person height
            torso_heights = []
            torso_heights.append(np.linalg.norm(openpose[9] - openpose[2]))
            torso_heights.append(np.linalg.norm(openpose[12] - openpose[5]))
            # Make torso 1/3 of the frame
            scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
            
            imgnames_.append(img_path)
            scales_.append(scale)
            centers_.append(center)
            part_.append(part)
                    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'{action}_real_{split}.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    print(out_file)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=part_
            )


def create_polevault_or_highjump_dataset(out_path, action="polevault", split='train'):
    if action in ["polevault", "highjump"]:
        file = files[1]
        # cgt: pole vault, tg: high jump, ymq: badminton
        action_label = "cgt" if action == "polevault" else "tg"
        annot_path = os.path.join(base_path, 'annotations', file)
        with open(annot_path) as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    
    # parts_ is the 2D keypoints. 
    imgnames_, scales_, centers_, part_  = [], [], [], []

    valid_videos = [vid['action_labels'] for vid in data if action_label in vid['action_labels']]
    train_len = 2  # This should match 
    print(f"Total {len(valid_videos)} videos. Using {train_len} for training.")
    if split == 'train':
        valid_labels = valid_videos[:train_len]
    elif split == 'test':
        valid_labels = valid_videos[train_len:]

    for vid in data:
        if vid['action_labels'] not in valid_labels:
            continue

        for meta in vid['frames'][:-5]: # exclude the last few frames because they are usually blurry
            img_path = os.path.join(base_path, 'images', meta['img_name'])
            openpose = np.array(meta['joints'])
            openpose[:, 2] = 1 - openpose[:, 2]  # 0 is visible.
            openpose[15:, 2] = 0.

            part = np.zeros([24, 3])
            part_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23])
            part[part_idx, :] = openpose[[11,10,9,12,13,14,4,3,2,5,6,7,0,16,15,18,17]]

            bbox = [min(openpose[:,0]), min(openpose[:,1]),
                    max(openpose[:,0]), max(openpose[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            
            # scaleFactor depends on the person height
            torso_heights = []
            torso_heights.append(np.linalg.norm(openpose[9] - openpose[2]))
            torso_heights.append(np.linalg.norm(openpose[12] - openpose[5]))
            # Make torso 1/3 of the frame
            scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
            
            imgnames_.append(img_path)
            scales_.append(scale)
            centers_.append(center)
            part_.append(part)
                    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'{action}_real_{split}.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    print(out_file)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=part_
            )


def create_diving_dataset(out_path, split='train'):
    action = "diving"
    file = files[0]
    annot_path = os.path.join(base_path, 'annotations', file)
    with open(annot_path) as f:
        data = json.load(f)
    
    # parts_ is the 2D keypoints. 
    imgnames_, scales_, centers_, part_  = [], [], [], []

    train_len = 15
    print(f"Total {len(data)} videos. Using {train_len} for training.")
    if split == 'train':
        data = data[:train_len]
    elif split == 'test':
        data = data[train_len:]

    for vid in data:
        for meta in vid['frames'][::10]:  # downsample.
            if len(meta['joints']) == 0: continue
            img_path = os.path.join(base_path, 'images', meta['img_name'])
            joints = meta['joints']
            openpose = np.array(joints)
            openpose[:, 2] = openpose[:, 2] != 2  # 0 is visible.
            # openpose[15:, 2] = 0.
            if np.sum(openpose[:, 2] > 0) < 4: continue

            part = np.zeros([24, 3])
            part_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23])
            part[part_idx, :] = openpose[[11,10,9,12,13,14,4,3,2,5,6,7,0,16,15,18,17]]

            bbox = [min(openpose[openpose[:,2] > 0,0]), min(openpose[openpose[:,2] > 0,1]),
                    max(openpose[openpose[:,2] > 0,0]), max(openpose[openpose[:,2] > 0,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            
            # scaleFactor depends on the person height
            torso_heights = []
            if openpose[9, 2] == 1 and openpose[2, 2] == 1:
                torso_heights.append(np.linalg.norm(openpose[9] - openpose[2]))
            if openpose[12, 2] == 1 and openpose[5, 2] == 1:
                torso_heights.append(np.linalg.norm(openpose[12] - openpose[5]))
            if torso_heights == []:
                mean_torso = 0.
            else:
                mean_torso = np.mean(torso_heights)
            
            if mean_torso == 0.: continue  # these usually just legs above the water.

            # Make torso 1/3 of the frame
            scale = max(mean_torso * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
            if scale < 0.5: continue  # too small.

            imgnames_.append(img_path)
            scales_.append(scale)
            centers_.append(center)
            part_.append(part)
                    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'{action}_real_{split}.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    print(out_file)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=part_
            )


if __name__ == "__main__":
    out_path = "./external/DAPA_release/data/dataset_extras"
    create_polevault_or_highjump_dataset(out_path, action='polevault', split='train')
    create_polevault_or_highjump_dataset(out_path, action='polevault', split='test')
