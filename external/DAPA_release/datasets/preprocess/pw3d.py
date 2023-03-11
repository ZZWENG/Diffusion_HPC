import os
import cv2
import numpy as np
import pickle

split = 'train'
def pw3d_extract(dataset_path, out_path):

    # scale factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_, parts_ = [], [], [], []
    poses_, shapes_, genders_ = [], [], [] 
    openpose_ = []

    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'train')
    files = [os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    # go through all the .pkl files
    for filename in files:
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            # get through all the people in the sequence
            for i in range(num_people):
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                gender = genders[i]
                # consider only valid frames
                for valid_i in range(valid_pose.shape[0]):
                    part = valid_keypoints_2d[valid_i,:,:].T
                    part = part[part[:,2]>0,:]
                    bbox = [min(part[:,0]), min(part[:,1]),
                        max(part[:,0]), max(part[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                    
                    # transform global pose
                    pose = valid_pose[valid_i]
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]                      
#                     import pdb; pdb.set_trace()
                    # read 2D detections
                    part25 = np.zeros([25,3])
                    part = valid_keypoints_2d[valid_i,:,:].T
                    part25[0, :] = part[0]  # nose
                    part25[1, :] = (part[2] + part[5]) / 2 # neck
                    part25[2, :] = part[2]  # r.shoulder
                    part25[5, :] = part[5]  # l.shoulder
                    part25[3, :] = part[3]  # r.elbow
                    part25[4, :] = part[4]  # r.wrist
                    part25[6, :] = part[6]  # l.elbow
                    part25[7, :] = part[7]  # l.wrist
                    part25[9, :] = part[8]  # r.hip
                    part25[10, :] = part[9]  # r.knee
                    part25[11, :] = part[10]  # r.ankle
                    part25[12, :] = part[11]  # l.hip
                    part25[13, :] = part[12]  # l.knee
                    part25[14, :] = part[13]  # l.ankle
                    part25[15, :] = part[14]  # r.eye
                    part25[16, :] = part[15]  # l.eye
                    part25[17, :] = part[16]
                    part25[18, :] = part[17]
                    
#                     part24[0, :] = part[10]  # r.ankle
#                     part24[1, :] = part[9]  # r.knee
#                     part24[2, :] = part[8]  # r.hip
#                     part24[3, :] = part[11]  # l.hip
#                     part24[4, :] = part[12]  # l.knee
#                     part24[5, :] = part[13]  # l.ankle
#                     part24[6, :] = part[4]  # r.wrist
#                     part24[7, :] = part[3]  # r.elbow
#                     part24[8, :] = part[2]  # r.shoulder
#                     part24[9, :] = part[5]  # l.shoulder
#                     part24[10, :] = part[6]  # l.elbow
#                     part24[11, :] = part[7]  # l.wrist
                    
#                     part24[19, :] = part[0]  # nose
#                     part24[20, :] = part[15]  # l.eye
#                     part24[21, :] = part[14]  # r.eye
#                     part24[22, :] = part[17]  # l.ear
#                     part24[23, :] = part[16]  # r.ear
                    
                    openpose_.append(part25)
                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)

    # store data
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        '3dpw_{}.npz'.format(split))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       pose=poses_,
                       shape=shapes_,
                       gender=genders_,
                       openpose=openpose_
            )
