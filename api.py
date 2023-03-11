"""
API for running our method on a single image.

Example usage:
    python api.py --text_prompt "a person is pole vaulting"

"""
import argparse, sys, os

import numpy as np
import PIL
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from diffusers import StableDiffusionPipeline

from d2i_pipeline import MyPipeline as StableDiffusionDepth2ImgPipeline
from generate_dataset import get_pipeline, get_depth_pipeline, get_random_latents, prepare_image, image_grid

device = 'cuda'
torch_dtype = torch.float16
img_size = 512
depth_size = 64
generator = torch.Generator(device=device).manual_seed(0)
pipe_depth2img = get_depth_pipeline(device)

def get_depth_pipeline(device):
    torch_dtype = torch.float16
    depth_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        revision="fp16",
        torch_dtype=torch_dtype,
        use_auth_token=False
    ).to(device)
    return depth_pipe


def get_pipeline(device, finetuned_on=None):
    torch_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4',
        revision="fp16",
        torch_dtype=torch_dtype,
        use_auth_token=False
    ).to(device)

    if finetuned_on == 'mpii':
        print('!!! Loading finetuned model on MPII !!!')
        pipe.unet.load_attn_procs('./data/sd_ft_mpii')
    elif finetuned_on == 'smart':
        print('!!! Loading finetuned model on SMART !!!')
        pipe.unet.load_attn_procs('./data/sd_ft_smart')
    else:
        assert finetuned_on is None, 'finetuned_on must be one of [None, "mpii", "smart"]'
    return pipe


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


def our_pipeline(pipe, text_prompt, gen_idx=0, real_image=None, depth=None,
                 hmr_method='bev', save_path=None, 
                 use_random_for_d2i=False, save_mesh=False, filter_by_vposer=False):
    """
    Args:
        text_prompt (str): the text prompt.
        gen_idx (int): used in parsing the name of the saved image.
        real_image (PIL.Image): the real image.
        hmr_method (str): the method for HMR.
        save_path (str): optional. path to save the result.
        use_random_for_d2i (bool): whether to use random human latent for depth2image model.
        save_mesh (bool): whether to save the mesh.
        filter_by_vposer (bool): whether to filter the generated images by vposer.

    Returns:
        PIL.Image: the final image.
    """
    # step 1: run text-conditioned Stable Diffusion.
    if real_image is None:
        init_image = pipe(
            [text_prompt],
            guidance_scale = 7.5,
            num_inference_steps = 50,
            num_images_per_prompt = 1,
        ).images[0]
    else:
        init_image = real_image

    # step 2: run mask rcnn on the image.
    init_image_np = np.array(init_image)
    outputs = predictor(init_image_np)
    v = Visualizer(init_image_np[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    is_person = (outputs["instances"].pred_classes == 0).cpu().numpy()
    is_person = np.where(is_person)[0]
    not_person = (outputs["instances"].pred_classes != 0).cpu().numpy()
    not_person = np.where(not_person)[0]

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    if len(is_person) == 0:
        print("MaskRCNN: No person is detected in the image.")
        mask = np.zeros_like(init_image_np)[:, :, 0]
        mask_image = PIL.Image.fromarray(mask)
        
    else:
        mask = np.zeros_like(outputs["instances"].pred_masks[0].cpu().numpy())
        for idx in is_person:
            mask += outputs["instances"].pred_masks[idx].cpu().numpy()
        mask = mask > 0
        mask_image = PIL.Image.fromarray(mask)
    
    non_person_mask = np.zeros_like(mask)
    if len(not_person) > 0:
        for idx in not_person:
            non_person_mask += outputs["instances"].pred_masks[idx].cpu().numpy()
        non_person_mask = non_person_mask > 0
    non_person_mask_image = PIL.Image.fromarray(non_person_mask)

    if depth is None:
        # step 3: run HMR on the image.
        if hmr_method == 'bev':
            from bev_utils import bev_single_inference
            if save_mesh:
                hmr_outputs = bev_single_inference(
                    init_image_np, save_mesh=save_mesh, 
                    mesh_name=os.path.join(save_path, f'mesh{gen_idx}.ply'))
            else:
                hmr_outputs = bev_single_inference(init_image_np, 
                                               save_mesh=False)

            if hmr_outputs is None:
                print("BEV: No person is detected in the image.")
                return None

            depth = hmr_outputs['rendered_image']
            depth_image = PIL.Image.fromarray(depth).convert('L')
            depth_map_small = depth_image.resize((depth_size, depth_size))

        else:
            dapa_path = './external/DAPA_release'
            sys.path.append(dapa_path)
            from hmr_utils import SPIN_wrapper
            runner = SPIN_wrapper('external/DAPA_release/data/model_checkpoint.pt', 'cuda')

            img, depth = runner.inference_single(init_image, mask_image, move_person_to_center=True)
            depth_image = PIL.Image.fromarray(depth.cpu().numpy())
            depth_map_small = depth_image.resize((depth_size, depth_size))

    else:
        depth_image = PIL.Image.fromarray(depth).resize((mask.shape[0], mask.shape[1]))
        depth_map_small = depth_image.resize((depth_size, depth_size))
        hmr_outputs = None

    # step 4: run depth2img on the image.
    mask = mask_image.resize((depth_size, depth_size))
    non_person_mask_small = non_person_mask_image.resize((depth_size, depth_size))
    depth_map = torch.from_numpy(np.array(depth_image)).to(device=pipe.device, dtype=torch_dtype).unsqueeze(0)
    image = init_image.resize((img_size, img_size))
    mask = np.array(mask) / 255.
    non_person_mask = np.array(non_person_mask_image).astype(np.float32)
    non_person_mask_small = np.array(non_person_mask_small)

    # merge with depth map.
    depth_map = (depth_map.max() + 50 - depth_map ) * (depth_map > 0).float()
    foreground = (np.array(depth_map_small) > 0).astype(np.float32)
    mask = ((mask + foreground) > 0).astype(np.float32)

    # keep non-person objects in the background
    mask = mask * (1 - non_person_mask_small)
    # take out non-person objects from the foreground
    depth_map[0][non_person_mask > 0] = 0

    mask = torch.from_numpy(mask).to(device=pipe.device, dtype=torch_dtype).unsqueeze(0).unsqueeze(0)
        
    random_latents = get_random_latents(generator, device)
    image_in = prepare_image(image).to(device=device, dtype=torch_dtype)  # [1, 3, 512, 512]
    image_latents = pipe.vae.encode(image_in).latent_dist.sample(generator=generator)
    image_latents *= pipe.scheduler.init_noise_sigma * 0.18215  # magic number

    combined_latents = image_latents
    if use_random_for_d2i:
        # use random human latents for depth2img
        combined_latents = image_latents * (1 - mask) + random_latents * mask

    n_prompt = "deformed, bad anatomy"
    images, _ = pipe_depth2img(
        prompt=text_prompt,
        image=image,
        latents=combined_latents,
        guidance_scale = 7.5,
        num_inference_steps = 50,
        depth_map = depth_map,
        negative_prompt=n_prompt,
        strength=0.7,
        generator=generator
    )
    final_image = images.images[0]
    depth_image = PIL.Image.fromarray(depth_map[0].cpu().numpy()).convert('L')
    vis_results = [
        init_image, 
        PIL.Image.fromarray(out.get_image()[:, :, ::-1]).resize((img_size, img_size)), 
        depth_image.resize((img_size, img_size)),
        final_image]

    if hmr_outputs is not None:
        vp_score = int(hmr_outputs['vposer_score'])
        vp_score_thres = 30
        if filter_by_vposer and vp_score < vp_score_thres:
            return None

    if save_path:
        grid = image_grid(vis_results, 1, len(vis_results))
        if vp_score < vp_score_thres:
            grid.save(os.path.join(save_path, f'easy_image{gen_idx}_vp_{vp_score}.png'))
        else:
            grid.save(os.path.join(save_path, f'hard_image{gen_idx}_vp_{vp_score}.png'))
    return init_image, final_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_prompt', type=str, required=False)
    parser.add_argument('--real_image', type=str, required=False, help='optional real image to use as a guide.')
    parser.add_argument('--output_path', type=str, required=False, default='./', help='optional path to save output image.')
    parser.add_argument('--hmr_method', type=str, required=False, default='bev', choices=['spin', 'bev'], 
                        help='hmr method to use. (bev or spin).')
    parser.add_argument('--num_images', type=int, required=False, default=20, help='number of images to generate.')
    parser.add_argument('--save_mesh', action='store_true', help='save mesh for each image. (default: False)')
    parser.add_argument('--use_random_for_d2i', action='store_true', help='use random human latents for depth2img. (default: False)')
    parser.add_argument('--num_cycles', type=int, required=False, default=1, help='number of cycles to run. (default: 1)')
    args = parser.parse_args()

    pipe = get_pipeline(device, finetuned_on=None)
    use_random_for_d2i = args.use_random_for_d2i

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # generate images.
    for idx in range(args.num_images):
        if args.num_cycles == 1:
            our_pipeline(pipe, args.text_prompt, idx+args.num_images, args.real_image, 
                        hmr_method=args.hmr_method, use_random_for_d2i=use_random_for_d2i,
                        save_path=args.output_path, save_mesh=args.save_mesh)

        else:
            _, image = our_pipeline(pipe, args.text_prompt, idx, args.real_image, 
                                hmr_method=args.hmr_method, use_random_for_d2i=use_random_for_d2i,
                                save_path=None, save_mesh=None)
            
            for iter in range(args.num_cycles - 1):
                _, image = our_pipeline(pipe, args.text_prompt, idx, image, 
                                    hmr_method=args.hmr_method, save_path=args.output_path, 
                                    use_random_for_d2i=use_random_for_d2i, save_mesh=args.save_mesh)
