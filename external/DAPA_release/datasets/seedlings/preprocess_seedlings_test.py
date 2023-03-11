import os
import argparse
import numpy as np
import scipy.io as sio
import pandas as pd
import config

# mapping from the annotated kp name to index in openpose-25 kp.
anno_kp_map = {
    'R.eye': 21, 'L.eye': 20,
    'R.shoulder': 8, 'L.shoulder': 9,
    'R.elbow': 7, 'L.elbow': 10,
    'R.wrist': 6, 'L.wrist': 11,
    'R.hip': 2, 'L.hip': 3,
    'R.knee': 1, 'L.knee': 4,
    'R.ankle': 0, 'L.ankle': 5,
    # additional names to account for RA typos
    'L..eye': 20, 'Leye': 20, 'R.Eye': 21
}


# additional knee and ankle keypoints annotated.
knee_ankle_files = [
    'seedlings_test_knees_ankles_p1.csv',
    'seedlings_test_knees_ankles_p2.csv'
]
knee_ankle_anns0 = pd.read_csv(knee_ankle_files[0], header=None)
knee_ankle_anns1 = pd.read_csv(knee_ankle_files[1], header=None)
knee_ankle_anns = knee_ankle_anns0.append(knee_ankle_anns1)


def parse_ann(df, prefix):
    indices = df[0].str.strip(prefix).map(anno_kp_map).values
    out = np.zeros((24, 3), dtype=np.float32)
    for index, values in zip(indices, df.values):
        try:
            out[index, 0] = values[1]
            out[index, 1] = values[2]
            out[index, 2] = 1. # meaning this keypoint is annotated
        except:
            print(df[0].str.strip(prefix))
    return out


def merge_two_anns(keypoints1, keypoints2):
    # merge two annotators keypoint annotation
    out = np.zeros((24, 3), dtype=np.float32)
    for i in range(24):
        if keypoints1[i, 2] == 0 and keypoints2[i, 2] == 0:
            continue
        if keypoints1[i, 2] == 0:
            out[i] = keypoints2[i]
        elif keypoints2[i, 2] == 0:
            out[i] = keypoints1[i]
        else:
            assert keypoints1[i,2] == 1 and keypoints2[i,2] == 1
            out[i] = (keypoints1[i] + keypoints2[i])/2
    return out



def seedlings_dataset_extract(dataset_path, out_path, extract_adult, extract_infant):
    months = range(6, 15)
    scaleFactor = 1.2  # bbox expansion factor
    
    assert(extract_adult or extract_infant)
    # TODO: consider adults for now.
    
    # parts_ is the ground truth 2D keypoints
    imgnames_, scales_, centers_, parts_  = [], [], [], []
    kp_path = r'data/seedlings/kp_path'
    filtere_num = 0
    for month in months:
        base_folder = os.path.join(dataset_path, 'month{}'.format(month))
        annotations_KR = pd.read_csv(os.path.join(kp_path, 'month{}_KR.csv'.format(month)), header=None)
        annotations_KO = pd.read_csv(os.path.join(kp_path, 'month{}_KO.csv'.format(month)), header=None)
        frames_list = np.unique(annotations_KO[3])
        for frame in frames_list:
            recording = frame.split('_')[0]
            frame_path = os.path.join(base_folder, '{}_{:02d}_fps3/images'.format(recording, month), frame.split('_')[1])
                                      
            keypoints_KR = annotations_KR[annotations_KR[3] == frame]
            keypoints_KO = annotations_KO[annotations_KO[3] == frame]
            
            if extract_adult:
                adult_KR = parse_ann(keypoints_KR[keypoints_KR[0].str.startswith('a.')], prefix='a.')
                adult_KO = parse_ann(keypoints_KO[keypoints_KO[0].str.startswith('a.')], prefix='a.')
                adult = merge_two_anns(adult_KR, adult_KO)
                
                if adult[:, 2].sum() > 1: # need at least two keypoints to compute a bounding box
                    if len(adult[adult[:, 2]>0.1]) < 6: 
                        filtere_num += 1
                        continue      
                    bbox = [min(adult[adult[:,2]==1,0]), min(adult[adult[:,2]==1,1]),
                            max(adult[adult[:,2]==1,0]), max(adult[adult[:,2]==1,1])]

                    if abs(bbox[2]-bbox[0]) < 200 and abs(bbox[3]-bbox[1]) < 200: 
                        filtere_num += 1
                        continue
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    
                                        # scaleFactor depends on the person height
                    torso_heights = []  # torso points: 8 (rshould), 2 (rhip), 9 (lshould), 3 (lhip)
                    if adult[8, 2]>0.5 and adult[2, 2]>0.5:
                        torso_heights.append(
                            np.linalg.norm(adult[8,:2] - adult[2,:2]))
                    if adult[9, 2]>0.5 and adult[3, 2]>0.5:
                        torso_heights.append(
                            np.linalg.norm(adult[9,:2] - adult[3,:2]))
                    if len(torso_heights)  == 0: 
                        filtere_num += 1
                        continue
                        
                        
                    # add knee and ankle keypoints from new annotations.
                    key = '{}_{:02d}_{}'.format(recording, month, frame.split('_')[1])
                    knee_ankle_kps = knee_ankle_anns[knee_ankle_anns[3] == key]
                    knee_ankle_kps = parse_ann(knee_ankle_kps[knee_ankle_kps[0].str.startswith('a.')], prefix='a.')
#                     import pdb; pdb.set_trace()
                    # overwrite RA's knee-ankle annotation with ours
                    for idx in [0,1,4,5]:
                        if knee_ankle_kps[idx, 2] == 1:
                            adult[idx] = knee_ankle_kps[idx]
                    
                    # Make torso 1/3 of the frame
                    scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200
#                     scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                    
                    imgnames_.append(frame_path)  
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(adult)
                    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'seedlings_dataset_new.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_)         
    
if __name__ == '__main__':
    seedlings_dataset_extract(r'data/seedlings', config.DATASET_NPZ_PATH, True, False)
