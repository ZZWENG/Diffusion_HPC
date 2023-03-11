import argparse, os, pickle

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('./external/DAPA_release')
import config
from models import SMPL
from utils.renderer import Renderer

from api import our_pipeline, get_pipeline
from evaluation.smart_quant_eval import get_text_prompt
from vis_utils import overlay_kp

device = 'cpu'
BBOX_IMG_RES = 224
smpl_model = SMPL(
    os.path.join('./external/DAPA_release', config.SMPL_MODEL_DIR), 
    batch_size=1, create_transl=False).to(device)
renderer = Renderer(faces = smpl_model.faces)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smart_base_path', type=str, default='./data/SportsCap_Dataset_SMART_v1')
    parser.add_argument('--use_text', action='store_true', default=False)
    parser.add_argument('--use_real', action='store_true', default=False)
    parser.add_argument('--save_smart', action='store_true', default=False)
    parser.add_argument('--out_path', type=str, default='./eval_data')
    parser.add_argument('--finetuned_on', type=str, default=None, choices=['smart', None], 
                        help='Dataset that SD was finetuned on.')
    parser.add_argument('--debug', action='store_true', default=False)

    return parser.parse_args()


def generate(args):
    pipe = get_pipeline('cuda', finetuned_on=args.finetuned_on)

    prefix = "ours_ft" if args.finetuned_on == 'smart' else "ours"
    if not args.use_text and args.use_real:
        out_path = os.path.join(args.out_path, 'quant_pose_eval_smart', f'{prefix}_notext')
    elif not args.use_real and args.use_text:
        out_path = os.path.join(args.out_path, 'quant_pose_eval_smart', f'{prefix}_nor')
    elif args.use_real and args.use_text:
        out_path = os.path.join(args.out_path, 'quant_pose_eval_smart', f'{prefix}')
    else:
        raise ValueError("Must use at least one of text or real.")
    
    out_smart_path = os.path.join(args.out_path, 'smart')
    if args.save_smart:
        if not os.path.exists(out_smart_path):
            os.makedirs(out_smart_path)
            
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for action in ['balancebeam', 'highjump', 'diving', 'polevault', 'vault', 'unevenbars']:
        generate_single_action(args, action, pipe, out_path, out_smart_path)


def generate_single_action(args, action, pipe, out_path, out_smart_path):
    print(f"Processing {action}")

    text = get_text_prompt(action)
    eft_base_path = os.path.join(args.smart_base_path, f'eft_smart/{action}')
    eft_data_all = os.listdir(eft_base_path)

    for idx, eft_file in enumerate(tqdm(eft_data_all)):
        with open(os.path.join(eft_base_path, eft_file), 'rb') as f:
            eft_data = pickle.load(f)

        imgName = eft_data['imageName'][0]
        imgFullPath = imgName
        # print(f'Input image: {imgFullPath}')

        if os.path.exists(imgFullPath) ==False:
            print(f"Img path is not valid: {imgFullPath}")
            assert False
        rawImg = Image.open(imgFullPath).convert('RGB')

        #EFT data
        bbox_scale = eft_data['scale'][0]
        bbox_center = eft_data['center'][0]
        bbox_size = bbox_scale * 200
        bbox = (bbox_center[0]-bbox_size/2, bbox_center[1]-bbox_size/2, bbox_center[0]+bbox_size/2, bbox_center[1]+bbox_size/2)
        crop = rawImg.crop(bbox)
        orig_kp = np.array(eft_data['keypoint2d'][0])        

        # turn coordinates from image space to bbox space
        box_kp = orig_kp.copy()
        box_kp[:, 0] = (orig_kp[:, 0] - bbox_center[0] + bbox_size / 2) / bbox_size * BBOX_IMG_RES
        box_kp[:, 1] = (orig_kp[:, 1] - bbox_center[1] + bbox_size / 2) / bbox_size * BBOX_IMG_RES

        pred_camera = np.array(eft_data['pred_camera'])    #(1,3)
        pred_betas = np.array( eft_data['pred_shape'], dtype=np.float32)    #(10,)
        pred_betas = torch.from_numpy(pred_betas).to(device)

        pred_pose_rotmat = np.reshape( np.array( eft_data['pred_pose_rotmat'], dtype=np.float32), (1,24,3,3)  )        #(24,3,3)
        pred_pose_rotmat = torch.from_numpy(pred_pose_rotmat).to(device)
        
        body_out = smpl_model(
            betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,:1], pose2rot=False)
        vertices = body_out.vertices

        cam = np.stack([pred_camera[:,1], pred_camera[:,2], 2*5000/(224 * pred_camera[:,0] +1e-9)], -1)

        rend_depth = torch.from_numpy(renderer(
            vertices[0].cpu().numpy(), cam[0], None, depth_only=True)).float()

        if args.save_smart:
            crop.resize((BBOX_IMG_RES, BBOX_IMG_RES)).save(os.path.join(out_smart_path, f"{action}_image_{idx}.png"))

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
            save_name = f"{action}_image_{idx}.png"  # Save in this format
            print('Saved to {}'.format(save_name))
            final_image.save(os.path.join(out_path, save_name))


if __name__ == '__main__':
    args = parse_args()
    generate(args)
