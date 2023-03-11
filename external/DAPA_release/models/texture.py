import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import trimesh

import models.net_blocks as nb
import config

def get_spherical_coords(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    theta = np.arccos(X[:, 2] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 1], X[:, 0])

    # Normalize both to be between [-1, 1]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv],1)

def compute_uvsampler(verts, faces, tex_size=2):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    import itertools
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])
    vs = verts[faces]
    # Compute alpha, beta (this is the same order as NMR)
    v2 = vs[:, 2]
    v0v2 = vs[:, 0] - vs[:, 2]
    v1v2 = vs[:, 1] - vs[:, 2]    
    # F x 3 x T*2
    samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)    
    # F x T*2 x 3 points on the sphere 
    samples = np.transpose(samples, (0, 2, 1))
    # Now convert these to uv.
    uv = get_spherical_coords(samples.reshape(-1, 3))
    # uv = uv.reshape(-1, len(coords), 2)

    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv


# texture_pred = self.texture_predictor.forward(img_feat)
# tex_loss = texture_loss(texture_pred, imgs, mask_pred, masks)

def get_texture_models():
    texture_model = TexturePredictorUV().cuda()
    texture_loss = PerceptualTextureLoss()
    
    print("==> T: Total parameters: {:.2f}M".format(sum(p.numel() for p in texture_model.parameters()) / 1000000.0))
    
    optimizer = torch.optim.Adam(texture_model.parameters(), lr=1e-4)
    return {'model': texture_model, 'optimizer': optimizer, 'loss': texture_loss, 'dt_loss': texture_dt_loss}
    

class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """
    def __init__(self, nz_feat=2048, batch_size=64, n_upconv=5, nc_init=256, predict_flow=True, symmetric=False):
        super(TexturePredictorUV, self).__init__()
        tex_size = 2
        # compute uv_sampler
        mesh = trimesh.load(config.UV_MESH_FILE)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        num_verts = verts.shape[0]

        self.mean_v = nn.Parameter(torch.Tensor(verts))
        self.num_output = num_verts
        num_faces = faces.shape[0]
        
        verts_np = verts
        faces_np = faces
      
        uv_sampler = compute_uvsampler(verts_np, faces_np[:num_faces], tex_size=tex_size)
        # F' x T x T x 2
        uv_sampler = Variable(torch.FloatTensor(uv_sampler).cuda(), requires_grad=False)
        # B x F' x T x T x 2
        uv_sampler = uv_sampler.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        print(uv_sampler.shape)
        img_H = int(2**np.floor(np.log2(np.sqrt(num_faces) * tex_size)))
        img_W = img_H # * 2
        nb.net_init(self)
        
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

        self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2)
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=nc_final, use_deconv=False, upconv_mode='bilinear')

    def forward(self, feat):
        uvimage_pred = self.enc.forward(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder.forward(uvimage_pred)
        self.uvimage_pred = torch.nn.functional.tanh(self.uvimage_pred)
        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)
        # Contiguous Needed after the permute..
        return tex_pred.contiguous()
        
        
def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            #n = m.out_features
            #m.weight.data.normal_(0, 0.02 / n) #this modified initialization seems to work better, but it's very hacky
            #n = m.in_features
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #xavier
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv2d): #or isinstance(m, nn.ConvTranspose2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #this modified initialization seems to work better, but it's very hacky
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            # Initialize Deconv with bilinear weights.
            base_weights = bilinear_init(m.weight.data.size(-1))
            base_weights = base_weights.unsqueeze(0).unsqueeze(0)
            m.weight.data = base_weights.repeat(m.weight.data.size(0), m.weight.data.size(1), 1, 1)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n))
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            

def texture_dt_loss(texture_flow, dist_transf, vis_rend=None, cams=None, verts=None, tex_pred=None):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    dist_transf: B x 1 x N x N

    Similar to geom_utils.sample_textures
    But instead of sampling image, it samples dt values.
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 1 x F x T*T
    dist_transf = torch.nn.functional.grid_sample(dist_transf, flow_grid)

    if vis_rend is not None:
        # Visualize the error!
        # B x 3 x F x T*T
        dts = dist_transf.repeat(1, 3, 1, 1)
        # B x 3 x F x T x T
        dts = dts.view(-1, 3, F, T, T)
        # B x F x T x T x 3
        dts = dts.permute(0, 2, 3, 4, 1)
        dts = dts.unsqueeze(4).repeat(1, 1, 1, 1, T, 1) / dts.max()

#         from ..utils import bird_vis
#         for i in range(dist_transf.size(0)):
#             rend_dt = vis_rend(verts[i], cams[i], dts[i])
#             rend_img = bird_vis.tensor2im(tex_pred[i].data)            
#             import matplotlib.pyplot as plt
#             plt.ion()
#             fig=plt.figure(1)
#             plt.clf()
#             ax = fig.add_subplot(121)
#             ax.imshow(rend_dt)
#             ax = fig.add_subplot(122)
#             ax.imshow(rend_img)
#             import ipdb; ipdb.set_trace()

    return dist_transf.mean()


class PerceptualTextureLoss(object):
    def __init__(self):
#         from ..nnutils.perceptual_loss import PerceptualLoss
        self.perceptual_loss = PerceptualLoss()

    def __call__(self, img_pred, img_gt, mask_pred, mask_gt):
        """
        Input:
          img_pred, img_gt: B x 3 x H x W
        mask_pred, mask_gt: B x H x W
        """
        mask_pred = mask_pred.unsqueeze(1)
        mask_gt = mask_gt.unsqueeze(1)
        
        # debugging script
#         masked_rend = (img_pred * mask_pred)[0].data.cpu().numpy()
#         masked_gt = (img_gt * mask_gt)[0].data.cpu().numpy()
#         import matplotlib.pyplot as plt
#         plt.ion()
#         plt.figure(1)
#         plt.clf()
#         fig = plt.figure(1)
#         ax = fig.add_subplot(151)
#         ax.imshow(np.transpose(masked_rend, (1, 2, 0)))
#         ax = fig.add_subplot(152)
#         ax.imshow(np.transpose(masked_gt, (1, 2, 0)))
#         ax = fig.add_subplot(153)
#         ax.imshow(np.transpose(img_gt[0].data.cpu().numpy(), (1, 2, 0)))
#         ax = fig.add_subplot(154)
#         ax.imshow(np.transpose(mask_pred[0].data.cpu().numpy(), (1, 2, 0)))
#         ax = fig.add_subplot(155)
#         ax.imshow(np.transpose(mask_gt[0].data.cpu().numpy(), (1, 2, 0)))
#         import pdb; pdb.set_trace()
        
        # Only use mask_gt..
        dist = self.perceptual_loss(img_pred * mask_gt, img_gt * mask_gt)
        return dist.mean()
    
class PerceptualLoss(object):
    def __init__(self, model='net', net='alex', use_gpu=True):
        print('Setting up Perceptual loss..')
        from external.PerceptualSimilarity.models import dist_model

        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=True)
        print('Done')

    def __call__(self, pred, target, normalize=True):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        if 'forward_pair' in dir(self.model):
            dist = self.model.forward_pair(target, pred)
        else:
            dist = self.model.forward(target, pred)

        return dist