import argparse, os, json

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('./external/DAPA_release')
import config
from models import SMPL
from utils.renderer import Renderer

from api import our_pipeline, get_pipeline
from evaluation.mpii_quant_eval import generate_dataset_obj
from vis_utils import overlay_kp


device = 'cpu'
smpl_model = SMPL(
    os.path.join('./external/DAPA_release', config.SMPL_MODEL_DIR), 
    batch_size=1, create_transl=False).to(device)
renderer = Renderer(faces = smpl_model.faces)
BBOX_IMG_RES = 224

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpii_base_path', type=str, default='./data/mpii')
    parser.add_argument('--use_text', action='store_true', default=False)
    parser.add_argument('--use_real', action='store_true', default=False)
    parser.add_argument('--save_mpii', action='store_true', default=False)
    parser.add_argument('--out_path', type=str, default='./eval_data')
    parser.add_argument('--finetuned_on', type=str, default=None, choices=['mpii', None], 
                        help='Dataset that SD was finetuned on.')
    parser.add_argument('--debug', action='store_true', default=False)

    return parser.parse_args()


def parse_text(act):
    if act['act_id'] == -1:
        return ""
    text = '; '.join([act['cat_name'], act['act_name']])
    return text


def generate(args):
    # original MPII annotations.
    base_path = args.mpii_base_path
    mpii_anns_path = os.path.join(base_path, 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
    decoded1 = sio.loadmat(mpii_anns_path, struct_as_record=False)["RELEASE"]
    obj = generate_dataset_obj(decoded1)

    pipe = get_pipeline('cuda', finetuned_on=args.finetuned_on)

    jpg_name_to_idx = {}
    for i, item in enumerate(obj['annolist']):
        jpg_name_to_idx[item['image']['name']] = i

    eft_path = os.path.join(base_path, 'MPII_ver01.json')
    imgDir = os.path.join(base_path, 'images')
    with open(eft_path, 'r') as f:
        eft_data = json.load(f)
        print("EFT data: ver {}".format(eft_data['ver']))
        eft_data_all = eft_data['data']

    prefix = 'ours_ft' if args.finetuned_on == 'mpii' else 'ours'
    if not args.use_text and args.use_real:
        out_path = os.path.join(args.out_path, 'quant_pose_eval_mpii', f'{prefix}_notext')
    elif not args.use_real and args.use_text:
        out_path = os.path.join(args.out_path, 'quant_pose_eval_mpii', f'{prefix}_nor')
    elif args.use_real and args.use_text:
        out_path = os.path.join(args.out_path, 'quant_pose_eval_mpii', f'{prefix}')
    else:
        raise ValueError("Must use either text or real.")
    
    # if args.save_mpii, save the original MPII data.
    out_mpii_path = os.path.join(args.out_path, 'mpii')
    if args.save_mpii:
        if not os.path.exists(out_mpii_path):
            os.makedirs(out_mpii_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for idx, eft_data in enumerate(tqdm(eft_data_all)):
        #Get raw image path
        imgFullPath = eft_data['imageName']
        imgName = imgFullPath
        imgFullPath =os.path.join(imgDir, imgName)
        if os.path.exists(imgFullPath) ==False:
            print(f"Img path is not valid: {imgFullPath}")
            assert False
        rawImg = Image.open(imgFullPath).convert('RGB')

        #EFT data
        bbox_scale = eft_data['bbox_scale']
        bbox_center = eft_data['bbox_center']
        bbox_size = bbox_scale * 200
        bbox = (bbox_center[0]-bbox_size/2, bbox_center[1]-bbox_size/2, bbox_center[0]+bbox_size/2, bbox_center[1]+bbox_size/2)
        crop = rawImg.crop(bbox)
        orig_kp = np.array(eft_data['gt_keypoint_2d'])

        # turn coordinates from image space to bbox space
        box_kp = orig_kp.copy()
        box_kp[:, 0] = (orig_kp[:, 0] - bbox_center[0] + bbox_size / 2) / bbox_size * BBOX_IMG_RES
        box_kp[:, 1] = (orig_kp[:, 1] - bbox_center[1] + bbox_size / 2) / bbox_size * BBOX_IMG_RES

        pred_camera = np.array(eft_data['parm_cam'])[np.newaxis, :]     #(1,3)
        pred_betas = np.reshape(np.array( eft_data['parm_shape'], dtype=np.float32), (1,10) )     #(10,)
        pred_betas = torch.from_numpy(pred_betas).to(device)

        pred_pose_rotmat = np.reshape( np.array( eft_data['parm_pose'], dtype=np.float32), (1,24,3,3)  )        #(24,3,3)
        pred_pose_rotmat = torch.from_numpy(pred_pose_rotmat).to(device)
        
        body_out = smpl_model(
            betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,:1], pose2rot=False)
        vertices = body_out.vertices

        cam = np.stack([pred_camera[:,1], pred_camera[:,2], 2*5000/(224 * pred_camera[:,0] +1e-9)], -1)
        
        act = obj['act'][jpg_name_to_idx[imgName]]
        text = parse_text(act)
        if text == "": continue

        rend_depth = torch.from_numpy(renderer(
            vertices[0].cpu().numpy(), cam[0], None, depth_only=True)).float()

        if args.save_mpii:
            # Save the original MPII data
            out_mpii_path_img = os.path.join(out_mpii_path, imgName)
            crop.resize((BBOX_IMG_RES, BBOX_IMG_RES)).save(out_mpii_path_img)

        if args.debug:
            Image.fromarray(rend_depth.numpy()).convert('L').save('demo_depth.png')
            crop.save('bbox.jpg')
            overlay_kp(rawImg, orig_kp).save('demo_overlay.png')
            overlay_kp(crop.resize((BBOX_IMG_RES, BBOX_IMG_RES)), box_kp).save('crop_overlay.png')
            import pdb; pdb.set_trace()

        crop = crop.resize((BBOX_IMG_RES, BBOX_IMG_RES))
        if (not args.use_text) and args.use_real:
            ours_out = our_pipeline(
                pipe, "", 0, real_image=crop, depth=rend_depth.numpy(), filter_by_vposer=True)
        elif (not args.use_real) and args.use_text:
            ours_out = our_pipeline(
                pipe, text, 0, real_image=None, depth=rend_depth.numpy(), filter_by_vposer=True)
        elif args.use_real and args.use_text:
            ours_out = our_pipeline(
                pipe, text, 0, real_image=crop, depth=rend_depth.numpy(), filter_by_vposer=True)
        else:
            raise ValueError("Must use either text or real.")

        if ours_out is not None:
            _, final_image = ours_out
            final_image.save(os.path.join(out_path, imgName))


if __name__ == '__main__':
    args = parse_args()
    generate(args)
