import os, sys
import glob
import json
import numpy as np
import tqdm

import argparse
import cv2
import torch
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True, help='')
parser.add_argument('--out_path', required=True, help='')

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# whether to use DARKPOSE instead of OPENPOSE.
USE_DARKPOSE = False

# if USE_DARKPOSE: 
#     sys.path.insert(0, '/home/groups/syyeung/zzweng/code/DarkPose/lib')
#     from config import cfg
#     from config import update_config
#     from core.inference import get_max_preds, gaussian_blur, taylor
#     from models.pose_hrnet import get_pose_net
#     cfg.defrost()
#     cfg.merge_from_file('/home/groups/syyeung/zzweng/code/DarkPose/experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml')
#     cfg.freeze()
#     image_size = np.array(cfg.MODEL.IMAGE_SIZE)
#     model = get_pose_net(cfg, is_train=False).cuda().eval()
#     model_state_file = '/scratch/users/zzweng/w48_384×288.pth'
#     model.load_state_dict(torch.load(model_state_file))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []
    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        keypoints.append(body_keypoints)

    return keypoints


def extract(dataset_base, out_path):
    # parts_ is the 2D keypoints. 
    imgnames_, scales_, centers_, openpose_  = [], [], [], []
    scaleFactor = 1.2   # bbox expansion factor

    videos = glob.glob(os.path.join(dataset_base, '*.mkv')) + glob.glob(os.path.join(dataset_base, '*.mp4'))
    
    for video in videos:
        dataset_path = os.path.join(dataset_base, os.path.splitext(os.path.split(video)[1])[0])

        images_path = os.path.join(dataset_path, 'images')
        keypoints_path = os.path.join(dataset_path, 'keypoints')
        images = glob.glob(os.path.join(images_path, '*.jpg')) 
        images = sorted(images)
        print(len(images), 'images')

        for img_path in tqdm.tqdm(images):
            img_name = os.path.split(img_path)[1].strip(".jpg")
            kp_path = os.path.join(keypoints_path, img_name+'_keypoints.json')
            parts = read_keypoints(kp_path)
            for person_idx in range(len(parts)):
                part = parts[person_idx]
                bbox = [min(part[part[:,2]>0,0]), min(part[part[:,2]>0,1]),
                                max(part[part[:,2]>0,0]), max(part[part[:,2]>0,1])]
                bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                if part[:, 2].sum() < 5.: continue

                # scaleFactor depends on the person height
                torso_heights = []  # torso points: 2 (rshould), 9 (rhip), 5 (lshould), 12 (lhip)
                if part[2,2] > 0 and part[9,2] > 0:
                    torso_heights.append(np.linalg.norm(part[2,:2] - part[9,:2]))
                if part[5,2] > 0 and part[12,2] > 0:
                    torso_heights.append(np.linalg.norm(part[5,:2] - part[12,:2]))
                # Make torso 1/3 of the frame
                if len(torso_heights) == 0:
                    continue # the torso is not visible
                    scale = max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor / 200
                else:
                    scale = max(np.mean(torso_heights) * 3, max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * scaleFactor) / 200

                if scale * 200 < 5:
                    continue

                imgnames_.append(img_path)
                scales_.append(scale)
                centers_.append(center)

                # (Aug 20 experiment): Replace part with DarkPose results
                # if USE_DARKPOSE:
                #     image = cv2.imread(img_path)
                #     orig_height, orig_width = image.shape[:2]
                    
                #     box_width = bbox[2] - bbox[0]
                #     box_height = bbox[3] - bbox[1]
                #     cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1]
                #     # print(cropped_image.shape)
                #     cropped_image = cv2.resize(cropped_image, (384, 288))
                    
                #     input = transform(cropped_image).unsqueeze(0).cuda()
                #     output = model(input)
                #     hm = output.cpu().detach().numpy()
                #     coords, maxvals = get_max_preds(hm)
                #     heatmap_height = hm.shape[2]
                #     heatmap_width = hm.shape[3]

                #     # post-processing
                #     hm = gaussian_blur(hm, cfg.TEST.BLUR_KERNEL)
                #     hm = np.maximum(hm, 1e-10)
                #     hm = np.log(hm)
                #     for n in range(coords.shape[0]):
                #         for p in range(coords.shape[1]):
                #             coords[n,p] = taylor(hm[n][p], coords[n][p])

                #     preds = coords.copy()[0]  # (17, 2)
                #     # transform preds back to image coordinate
                #     preds[:, 0] *= (box_width/heatmap_width)
                #     preds[:, 1] *= (box_height/heatmap_height)
                #     preds[:, 0] += bbox[0]
                #     preds[:, 1] += bbox[1]
                #     darkpose = np.zeros((25, 3))
                #     darkpose[[0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11], :2] = preds
                #     darkpose[[0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11], 2] = 1.
                #     openpose_.append(darkpose)

                openpose_.append(part)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'PSBOjqCtpEU_fps3.npz')
    print('Preprocessed {} persons.'.format(len(imgnames_)))
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_, openpose=openpose_
            )
               
    
if __name__ == '__main__':
    args = parser.parse_args()
    extract(args.input_path, args.out_path)
