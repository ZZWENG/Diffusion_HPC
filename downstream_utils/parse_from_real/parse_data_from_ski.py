import json
import os
import numpy as np
import h5py
from tqdm import tqdm

data_root = './data/ski_3d'

scaleFactor = 1.2   # bbox expansion factor
joint_names_h36m = [
        'hip',  # 0
        'rhip',  # 1
        'rknee',  # 2
        'rankle',  # 3
        'lhip',  # 4
        'lknee',  # 5
        'lankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
    ]

def create_ski_dataset(out_path, action='ski', num_train_samples=200, split='train', debug=False):
    h5_label_file = h5py.File(os.path.join(data_root, f'./{split}/labels.h5'), 'r')
    print('Available labels:',list(h5_label_file.keys()))

    # parts_ is the 2D keypoints. 
    imgnames_, scales_, centers_, part_, Ss_  = [], [], [], [], []

    num_samples = len(h5_label_file['seq'])
    if split == 'train':
        if num_train_samples > 0:
            num_samples = num_train_samples  # few shot finetuning

    for index in tqdm(range(num_samples)):
        seq   = int(h5_label_file['seq'][index])
        cam   = int(h5_label_file['cam'][index])
        frame = int(h5_label_file['frame'][index])
        subj  = int(h5_label_file['subj'][index])
        pose_3D = h5_label_file['3D'][index].reshape([-1,3])
        pose_2D = h5_label_file['2D'][index].reshape([-1,2]) # in range 0..1
        pose_2D = 256*pose_2D # in pixels, range 0..255

        # load image
        img_path = os.path.join(data_root, './{}/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(split,seq,cam,frame))

        # prepare 2D keypoints
        openpose = np.zeros([25, 3])
        openpose[[12, 13, 14, 9, 10, 11, 1, 5, 6, 7, 2, 3, 4], :2] = pose_2D[[4, 5, 6, 1, 2, 3, 8, 11, 12, 13, 14, 15, 16]]
        openpose[[12, 13, 14, 9, 10, 11, 1, 5, 6, 7, 2, 3, 4], 2] = 1
        part = np.zeros([24, 3])
        part_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23])
        part[part_idx, :] = openpose[[11,10,9,12,13,14,4,3,2,5,6,7,0,16,15,18,17]]

        # prepare 3D keypoints
        S17 = pose_3D
        S17 -= S17[0] # root-centered
        # Ski dataset's L/R leg joints are flipped
        r_leg = S17[[1, 2, 3]]
        l_leg = S17[[4, 5, 6]]
        S17[[1, 2, 3]] = l_leg
        S17[[4, 5, 6]] = r_leg
        # append 1 to last dim
        S17 = np.concatenate([S17, np.ones([17,1])], axis=1)

        if debug:
            from PIL import Image
            from vis_utils import overlay_kp, plot_3d_joints
            debug_image = Image.open(img_path)
            debug_image = overlay_kp(debug_image, openpose)
            debug_image.save('debug_2d.png')
            
            debug_image_3d = Image.open(img_path)
            debug_image_3d = plot_3d_joints(debug_image_3d, S17)
            debug_image_3d.save('debug_3d.png')
            import pdb; pdb.set_trace()

        bbox = [min(part[:,0]), min(part[:,1]),
                max(part[:,0]), max(part[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        
        # scaleFactor depends on the person height
        torso_heights = []
        torso_heights.append(np.linalg.norm(part[8] - part[2]))
        torso_heights.append(np.linalg.norm(part[9] - part[3]))
        # Make torso 1/3 of the frame
        scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
        
        imgnames_.append(img_path)
        scales_.append(scale)
        centers_.append(center)
        part_.append(part)
        Ss_.append(S17)
                    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'{action}_real_{split}.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    print('Saved to', out_file)
    if split == 'test':
        np.savez(out_file, imgname=imgnames_,
                        center=centers_,
                        scale=scales_,
                        part=part_,
                        S=Ss_)
    else:
        # do not use 3D keypoints for training
        np.savez(out_file, imgname=imgnames_,
                        center=centers_,
                        scale=scales_,
                        part=part_)


if __name__ == "__main__":
    out_path = "./external/DAPA_release/data/dataset_extras"
    # create_ski_dataset(out_path, split='train', debug=False)
    create_ski_dataset(out_path, split='test', debug=False)