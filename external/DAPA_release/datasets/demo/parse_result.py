import os, sys

import numpy as np
import PIL.Image as pil_img
import torchvision.transforms as T
import torch, cv2
from torch.utils.data import DataLoader
from torchgeometry import rotation_matrix_to_angle_axis
from tqdm import tqdm
import smplx

import config
from datasets import BaseDataset
from models import hmr
sys.path.append('/home/groups/syyeung/zzweng/code/seedlings_visual/utils')
from opendr_renderer import SMPLRenderer


crop_size = 224
target_focal = 500
adult_bm = smplx.SMPL(model_path='/home/groups/syyeung/behavioral_coding/body_models/smplx', create_body_pose=False, 
                 age='adult', kid_template_path='/home/groups/syyeung/behavioral_coding/body_models/smpl/SMPL_NEUTRAL.pkl').cuda();
opendr_renderer = SMPLRenderer(face_path='/home/groups/syyeung/zzweng/code/seedlings_visual/utils/smpl_faces.npy')
faces = torch.from_numpy(adult_bm.faces.astype(np.int32)).cuda()

def load_model(checkpoint_path):
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(checkpoint_path)
    print('Loaded checkpoint at step count:', checkpoint['total_step_count'])
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval();
    model.to('cuda')
    return model


def get_original(cam, x, y, h, orig_img_width, orig_img_height):
    scale = crop_size / h
    undo_scale = 1. / scale

    flength = 500.
    curr_focal = flength * undo_scale # predicted focal length
    tz = flength / (0.5 * crop_size * cam[0])
    
    trans = np.hstack([cam[1:], tz])
    
    dx = (orig_img_width/2 - (x+h/2)); dy = orig_img_height/2 - (y+h/2)
    trans[2] /= (curr_focal/target_focal)  # first, scale in the axis
    new_tz = tz / (curr_focal/target_focal)
    trans[0] -= new_tz * dx / target_focal  # then, shift
    trans[1] -= new_tz * dy / target_focal  
    return trans

def get_result(pred_pose, pred_betas, pred_cam):
    """From the model's direct output, get the vertices (3D), joints (3D). Only supports single person.
    """
    pred_pose = pred_pose.unsqueeze(0)
    pred_betas = pred_betas.unsqueeze(0) 
    pred_cam = pred_cam.unsqueeze(0).clone()
    bm = adult_bm(global_orient=pred_pose[:, :3].float(), body_pose=pred_pose[:, 3:].float(), betas=pred_betas)
   
    verts = bm.vertices.detach()[0].detach().cpu().numpy()
    return verts


def render_image_dapa(verts, img_path, out_image_path, pred_cam, x, y, h):
    framename = img_path.split('/')[-1]
    out_img_fn = os.path.join(out_image_path, 'rendered_{}'.format(framename))
    if os.path.exists(out_img_fn):
        # Another person in this image has been rendered in a previous batch. Render on top of that.
        img = cv2.imread(out_img_fn)
    else:
        img = cv2.imread(img_path)
    img = img[:, :, ::-1]
    orig_img_width = img.shape[1]
    orig_img_height = img.shape[0]
    
    trans = get_original(pred_cam, x.item(), y.item(), h.item(),  orig_img_width, orig_img_height)
    verts = verts + trans

    cam_for_render = np.hstack([target_focal, np.array([orig_img_width/2, orig_img_height/2])])
    rend_img = opendr_renderer(verts, cam_for_render, img=img, color_id=1)  # (1080, 1920, 3)
    pimg = pil_img.fromarray(rend_img)
    pimg.save(out_img_fn)
    return 


def generate(dataloader, model, out_image_path):
    for _, batch in tqdm(enumerate(dataloader)):
        center = batch['center'][0]
        w = h = batch['scale'][0] * 200
        x = center[0] - h/2
        y = center[1] - h/2

        input_image = batch['img']
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(input_image.to('cuda'))

        pred_rotmat_hom = torch.cat(
            [pred_rotmat.view(-1, 3, 3),
            torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda').view(1, 3, 1).expand(1*24, -1, -1)
            ], dim=-1)
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(1, -1)
        pred_pose[torch.isnan(pred_pose)] = 0.0
        verts = get_result(pred_pose[0], pred_betas[0], pred_camera[0])
        render_image_dapa(verts, batch['imgname'][0], out_image_path, pred_camera[0].detach().cpu().numpy(), x, y, h)


if __name__ == '__main__':
    ours_path = '/scratch/groups/syyeung/zzweng/logs/Jan8_gymnastics/checkpoints/2022_01_08-19_40_16.pt'
    seedlings_path = r'/scratch/groups/syyeung/seedlings/dapa_checkpoints/38341926_2021_11_14-20_34_42.pt'
    spin_pt_path = '/home/groups/syyeung/zzweng/code/SPIN/data/model_checkpoint.pt'
    model = load_model(spin_pt_path)

    from utils.train_options import TrainOptions
    args = TrainOptions().parser.parse_args([
        "--name", "test", "--noise_factor", "0", "--rot_factor", "0", "--scale_factor", "0"])
    dataset = BaseDataset(args, 'gymnastics', is_train=False, ignore_3d=True, parse_random_bg=True)
    dataloader = DataLoader(dataset, batch_size=1)
    # out_image_path = r'/scratch/groups/syyeung/zzweng/dapa_demo/jazz_dance/test/rO3R9KWb4fk/out_dapa4'
    out_image_path = r'/scratch/groups/syyeung/zzweng/dapa_demo/gymnastics/test/PSBOjqCtpEU/out_spin'
    generate(dataloader, model, out_image_path)

