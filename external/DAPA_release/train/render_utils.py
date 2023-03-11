import cv2
import os
import numpy as np
import torch
import neural_renderer as nr
import config

from models import SMPL

def obj_fv(fname): # read vertices id in faces: (vv1,vv2,vv3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [int(i) for i in tmp[1:4]]
                res.append(v)
    return np.array(res, dtype=np.int) - 1 # obj index from 1


def obj_vt(fname): # read texture coordinates: (u,v)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('vt '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:3]]
                res.append(v)
    return np.array(res, dtype=np.float)


def obj_ft(fname): # read texture id in faces: (vt1,vt2,vt3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[1]) for i in tmp[1:4]]
                else:
                    raise(Exception("not a textured obj file"))
                res.append(v)
    return np.array(res, dtype=np.int) - 1 # obj index from 1


template_obj = config.UV_MESH_FILE
fv = obj_fv(template_obj)
ft = obj_ft(template_obj)
vt = obj_vt(template_obj)
vt[:, 1] = 1- vt[:, 1]


class UVRenderer():
    def __init__(self, focal_length=5000., render_res=224):
        # Parameters for rendering
        self.focal_length = focal_length
        self.render_res = render_res
        # We use Neural 3D mesh renderer for rendering masks and part segmentations
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=render_res,
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=False)
        self.faces = torch.from_numpy(SMPL(config.SMPL_MODEL_DIR).faces.astype(np.int32)).cuda()

    def load_surreal_textures(self):
        surreal_path = config.SURREAL_ROOT
        with open(os.path.join(surreal_path, 'textures/female_all.txt')) as f:
            female_texture_paths = f.readlines()
        with open(os.path.join(surreal_path, 'textures/male_all.txt')) as f:
            male_texture_paths = f.readlines()
        all_texture_paths = female_texture_paths + male_texture_paths
        all_texture_paths = [os.path.join(surreal_path, p.strip('\n')) for p in all_texture_paths]
        self.all_texture_paths = all_texture_paths
        self.surreal_path = surreal_path

    def sample_texture(self):
        if not hasattr(self, 'all_texture_paths'):
            self.load_surreal_textures()
        # sample random texture from SURREAL
        texture_path = os.path.join(self.surreal_path, np.random.choice(self.all_texture_paths))
        surreal_texture = cv2.imread(texture_path)[:,:,::-1]/255.
        uv_textures = []
        uv_dim = surreal_texture.shape[0] # 512
        for i in range(ft.shape[0]):
            u, v = vt[ft[i]].mean(0)
            u, v = int(u*uv_dim), int(v*uv_dim)  # map from [0,1] to [0,512]
            uv_textures.append(surreal_texture[v, u])
            
        uv_textures = torch.tensor(uv_textures)
        uv_textures = uv_textures[None, :, None, None, None, :].expand(-1, -1, 2, 2, 2, -1)
        return uv_textures

    def __call__(self, vertices, camera, texture=None, use_surreal=False):
        batch_size = vertices.shape[0]
        if use_surreal:
            texture = torch.cat([self.sample_texture() for _ in range(batch_size)]).cuda().float()  

        cam_t = torch.stack([camera[:,1], camera[:,2], 2*self.focal_length/(self.render_res * camera[:,0] +1e-9)],dim=-1)
        
        K = torch.eye(3, device=vertices.device)
        K[0,0] = self.focal_length 
        K[1,1] = self.focal_length 
        K[2,2] = 1
        K[0,2] = self.render_res / 2.
        K[1,2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)
        R = torch.eye(3, device=vertices.device)[None, :, :].expand(batch_size, -1, -1)
        faces = self.faces[None, :, :].expand(batch_size, -1, -1)
        rendered, _, mask =  self.neural_renderer(vertices, faces, textures=texture, K=K, R=R, t=cam_t.unsqueeze(1))
        return rendered, mask  # (bn, 3, 224, 224)
    
    
uv_renderer = UVRenderer()
