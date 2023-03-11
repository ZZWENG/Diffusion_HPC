import cv2
import os
from tqdm import tqdm
import torch
import trimesh
import smplx

import sys
sys.path.append('./bev')
from bev import BEV, bev_settings, ResultSaver, collect_frame_path

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

expr_dir = './data/V02_05'
vp, ps = load_model(expr_dir, model_code=VPoser,
                              remove_words_in_model_weights='vp_model.',
                              disable_grad=True)
vp = vp.to('cuda')

def run_bev(args, in_folder_name, out_folder_name, apply_pose_aug = True):
    
    in_path = os.path.join(args.out_path, in_folder_name)
    out_path = os.path.join(args.out_path, out_folder_name)
    bev_args = bev_settings([
        "-m", "video", 
        "--show_items", "mesh",
        "--renderer", "pyrender",
        "-i", in_path, 
        "-o", out_path])
    bev = BEV(bev_args)
    bev.settings.pose_aug = apply_pose_aug
    frame_paths, _ = collect_frame_path(bev_args.input, bev_args.save_path)
    save_npz_path = bev_args.save_path.replace("bev", "bev_npz")
    saver = ResultSaver(bev_args.mode, bev_args.save_path, save_npz_path, save_npz=True)
    for frame_path in tqdm(frame_paths):
        print(frame_path)
        image = cv2.imread(frame_path)
        if image is None: continue

        outputs = bev(image)
        saver(outputs, frame_path)


def bev_single_inference(image_np, save_mesh=False, mesh_name='mesh.ply'):
    bev_args = bev_settings([
        "-m", "video", 
        "--show_items", "mesh",
        "--renderer", "pyrender"])
    bev = BEV(bev_args)
    bev.settings.pose_aug = False
    outputs = bev(image_np)
    if save_mesh:
        SMPL_MODEL_PATH = './data/smpl'
        smpl = smplx.SMPL(SMPL_MODEL_PATH)
        faces = smpl.faces
        mesh = trimesh.Trimesh(vertices=outputs['verts'][0], faces=faces)
        mesh.export(mesh_name)

    if outputs is None:
        return None
    
    emb = vp.encode(torch.from_numpy(outputs['smpl_thetas'][:1, 3:-6]).cuda()).rsample()
    outputs['vposer_score'] = (emb**2).sum().item()
    return outputs  # ['rendered_image']: (512, 512) np array
