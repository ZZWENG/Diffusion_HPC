import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        self.color = (0.8, 0.3, 0.3, 1.0)
    
    def set_color(self, color):
        self.color = color

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2, padding=0)
        return rend_imgs
    
    def visualize_images(self, images_1, images_2):
        imgs = []
        for i in range(images_1.shape[0]):
            imgs.append(images_1[i])
            imgs.append(images_2[i])
        imgs = make_grid(imgs, nrow=2, padding=0)
        return imgs
    
    def visualize_tb_with_synthetic(self, opt_vertices, opt_camera_translation, pred_vertices, pred_camera_translation,
                                    gen_imgs, real_imgs):
        opt_vertices = opt_vertices.cpu().numpy()
        opt_camera_translation = opt_camera_translation.cpu().numpy()
        pred_vertices = pred_vertices.cpu().numpy()
        pred_camera_translation = pred_camera_translation.cpu().numpy()
        gen_imgs = gen_imgs.cpu()
        gen_imgs_np = np.transpose(gen_imgs.numpy(), (0,2,3,1))
        real_imgs = real_imgs.cpu()
        real_imgs_np = np.transpose(real_imgs.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(opt_vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(opt_vertices[i], opt_camera_translation[i], gen_imgs_np[i]), (2,0,1))).float()
            pred_img = torch.from_numpy(np.transpose(self.__call__(pred_vertices[i], pred_camera_translation[i], gen_imgs_np[i]), (2,0,1))).float()
            rend_imgs.append(real_imgs[i])
            rend_imgs.append(gen_imgs[i])
            rend_imgs.append(rend_img)
            rend_imgs.append(pred_img)
        rend_imgs = make_grid(rend_imgs, nrow=4, padding=0)
        return rend_imgs

    def visualize_rgba(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        assert vertices.shape[0]==1
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_img = torch.from_numpy(np.transpose(self.__call__(
            vertices[0], camera_translation[0], images_np[0], rgba=True), (2,0,1))).float()
        return rend_img
    
    def visualize_depth(self, vertices, camera_translation):
        vertices = vertices.cpu().numpy()
        assert vertices.shape[0]==1
        camera_translation = camera_translation.cpu().numpy()
        rend_img = torch.from_numpy(self.__call__(
            vertices[0], camera_translation[0], None, depth_only=True)).float()
        return rend_img

    def __call__(self, vertices, camera_translation, image, rgba=False, depth_only=False):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=self.color)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        
        if rgba:
            image = np.ones_like(color)
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
            return np.concatenate([output_img, valid_mask], 2)
        elif depth_only:
            return rend_depth
        else:
            output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img

    def render_multiperson_depth(self, batched_vertices, camera_translation, image, rgba=False, depth_only=False):
        """
        batched_vertices: [num_person, num_vertices, 3]
        camera_translation: [num_person, 3]
        image: [3, H, W]
        """
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=self.color)

        # camera_translation[0] *= -1.
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))

        num_person = batched_vertices.shape[0]
        for p in range(num_person):
            vertices = batched_vertices[p]
            vertices += camera_translation[p][None, :]  # offset vertices to camera translation

            mesh = trimesh.Trimesh(vertices, self.faces)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, f'mesh_{p}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation * 0
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        
        if rgba:
            image = np.ones_like(color)
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
            return np.concatenate([output_img, valid_mask], 2)
        elif depth_only:
            return rend_depth
        else:
            output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
