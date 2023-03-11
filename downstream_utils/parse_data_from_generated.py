"""Create dataset in format accepted by SPIN/DAPA.

Output file name has the format: <action>_syn_train.npz

"""

import os
import pickle
import numpy as np
import tqdm
import argparse

scaleFactor = 1.2   # bbox expansion factor
SMPL_24_to_OP_25 = np.array([
    24, 12, 17, 19, 21, 16, 18, 20, 49, 
    45, 5, 8, 46, 4, 7,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34])

def extract_grouns_truths(generation_path, iter, debug=False):
    smpl_path = os.path.join(generation_path, 'spin')
    images_path = os.path.join(generation_path, 'final_images')
    spin_gt_path = os.path.join(generation_path, 'spin_gt.pkl')

    if iter > 0:
        smpl_path = smpl_path + f'_iter{iter}'
        images_path = images_path + f'_iter{iter}'
        spin_gt_path = os.path.splitext(spin_gt_path)[0] + f'_iter{iter}.pkl'

    smpl_files = sorted(os.listdir(smpl_path))

    with open(os.path.join(generation_path, spin_gt_path), 'rb') as f:
        gt_data = pickle.load(f)

    # parts_ is the 2D keypoints. 
    imgnames_, scales_, centers_, parts_  = [], [], [], []
    poses_, shapes_ = [], []
    
    for full_file_name in tqdm.tqdm(smpl_files):
        file_name, file_ext = os.path.splitext(full_file_name)
        if file_name.endswith('_grid'): continue

        file_parts = file_name.split('_')
        image_fn = '_'.join(file_parts[:-1])
        sample_idx = int(file_parts[-1])

        image_name = os.path.splitext(image_fn)[0]
        if image_name.endswith('grid') or image_name.startswith('.'): continue

        pose = gt_data[full_file_name]['thetas'][0]
        beta = gt_data[full_file_name]['betas'][0]
        pred_kp = gt_data[full_file_name]['pj2d_org'][0, :25]
        openpose = (pred_kp / 224 * 512 + 512 / 2)
        
        part = np.zeros([24,3])
        part_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23])
        part[part_idx, :2] = openpose[[11,10,9,12,13,14,4,3,2,5,6,7,0,16,15,18,17]]
        part[part_idx, 2] = 1
        
        img_path = os.path.join(images_path, full_file_name)
        if not os.path.exists(img_path):
            print(f'{img_path} does not exist.')
            continue

        if debug:
            from PIL import Image
            from vis_utils import overlay_kp
            debug_image = Image.open(img_path)
            debug_image = overlay_kp(debug_image, openpose)
            debug_image.save(f'debug_{full_file_name}')
            import pdb; pdb.set_trace()
    
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

    return imgnames_, centers_, scales_, parts_, poses_, shapes_


def create_datasets(generation_path, iter, action_name, out_path, split='train', debug=False):
    imgnames_, centers_, scales_, parts_, poses_, shapes_ = extract_grouns_truths(
        generation_path, iter=iter, debug=debug)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, f'{action_name}_syn_{split}.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    print('Saving to {}'.format(out_file))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       pose=poses_,
                       shape=shapes_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_path', type=str, required=True, help='path to the generation folder')
    parser.add_argument('--iter', type=int, default=0, help='Extract images from the end of this iteration.')
    parser.add_argument('--action', type=str, default='ski')
    parser.add_argument('--out_path', type=str, default="./external/DAPA_release/data/dataset_extras")
    args = parser.parse_args()

    create_datasets(args.generation_path, args.iter, args.action, args.out_path, debug=False)

    