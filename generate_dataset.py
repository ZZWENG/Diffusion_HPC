"""
This script generates images from a pre-trained model.

Example usage:
    python generate_dataset.py --out_path ./gen_data --num_images 10 

This script will generate images from the pre-trained model and save them to the --out_path directory.
The output_path directory will have the following structure:
    output_path
    ├── init_images
    │   ├── 0.png
    │   ├── 1.png
    ├── masks
    ├── bev or spin
    ├── final_images
    ├── spin_gt.pkl
"""

import argparse
import random, os, sys
import pickle

import cv2
from loguru import logger
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
from img_utils import image_grid, prepare_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="the number of inference steps to take"
    )
    parser.add_argument(
        "--use_real_guide",
        action="store_true",
        help="whether to use real guide images. If so, the guide images should be in the --out_path/init_images directory."
    )
    parser.add_argument(
        "--real_guidance_action",
        type=str,
        default="ski",
        choices=["ski", "polevault", "highjump", "balancebeam", "vault", "unevenbars", "diving"],
    )
    parser.add_argument(
        "--rg_num_samples",
        type=int,
        default=200,
        help="the number of real guide images to use. [!!!Only used for ski.]"
    )
    parser.add_argument(
        "--use_random_latents_for_human",
        action="store_true",
        help="whether to use random latents for the human body."
    )
    parser.add_argument(
        "--apply_pose_aug",
        action="store_true",
        help="whether to apply pose augmentation."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="the number of synthetic images to generate per (real) guide."
    )
    parser.add_argument(
        "--hmr_method",
        type=str,
        default="spin",
        choices=["spin", "bev"],
        help="the method used for human pose estimation."
    )
    parser.add_argument(
        "--add_blur",
        action="store_true",
        help="whether to add blur to the generated images."
    )
    parser.add_argument(
        "--action",
        type=str,
        default="doing pole vaulting",
        help="the text that will be used in prompt template such as `a person {action}`."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./",
        required=True,
        help="the path to save the output image"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="the number of images to generate. Not used if --use_real_guide is True."
    )
    parser.add_argument(
        "--num_opt_cycles",
        type=int,
        default=0,
        help="the number of optimization cycles to run"
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
        help="the strenght for the depth2img generation phase"
    )

    args = parser.parse_args()
    return args

prompt_template = [
    'a picture of an athlete {}',
    'a photo of an athlete {}',
    'a low resolution picture of a person {}',
    'a blurry image of an athlete {}',
    'a nice shot of a person {}',
    'a side shot of a person {}',
    'a high speed shot of a person {}',
    'a slow motion shot of a person {}',
]

def sample_prompt(action):
    template_idx= random.choice(range(len(prompt_template)))
    prompt = prompt_template[template_idx].format(action)
    logger.info(f"Prompt: {prompt}")
    return template_idx, prompt


def generate_images(args, pipe, out_folder_name):
    logger.info("Generating images...")
    prompts = [sample_prompt(args.action) for _ in range(args.num_images)]
    out_path = os.path.join(args.out_path, out_folder_name)
    num_images = 1
    n_prompt = "deformed, bad anatomy"
    for i in range(0, len(prompts)):
        prompt_idx, prompt = prompts[i]
        images = pipe(
            [prompt],
            guidance_scale = 7.5,
            num_inference_steps = args.num_inference_steps,
            num_images_per_prompt = num_images,
            negative_prompt = n_prompt
        )
        best_image = images.images[0]    
        best_image.save(os.path.join(out_path, f"{i}.png"))


def run_maskrcnn(args, in_folder_name, out_folder_name, mask_out_best_person_only=False):
    logger.info("Running MaskRCNN...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    in_path = os.path.join(args.out_path, in_folder_name)
    out_path = os.path.join(args.out_path, out_folder_name)
    for filename in os.listdir(in_path):
        im = cv2.imread(os.path.join(in_path, filename))
        if im is None:
            continue
        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        is_person = (outputs["instances"].pred_classes == 0).cpu().numpy()
        is_person = np.where(is_person)[0]
        if len(is_person) == 0:
            # no person is detected.
            continue
        if mask_out_best_person_only:
            best_person_idx = np.argmax(outputs["instances"].scores[is_person].cpu().numpy())
            mask = outputs["instances"].pred_masks[best_person_idx].cpu().numpy()
        else:
            mask = np.zeros_like(outputs["instances"].pred_masks[0].cpu().numpy())
            for idx in is_person:
                mask += outputs["instances"].pred_masks[idx].cpu().numpy()
            mask = mask > 0
        PIL.Image.fromarray(mask).save(os.path.join(out_path, filename))
        # PIL.Image.fromarray(out.get_image()[:, :, ::-1]).save(os.path.join(out_path, 'vis_' + filename))


def get_random_latents(generator, device):
    latents = None
    seeds = []
    for _ in range(1):
        # Get a new random seed, store it and use it as the generator state
        seed = generator.seed()
        seeds.append(seed)
        generator = generator.manual_seed(seed)
        
        image_latents = torch.randn(
            (1, 4, 512 // 8, 512 // 8),
            generator = generator,
            device = device
        )
        latents = image_latents if latents is None else torch.cat((latents, image_latents))
        # latents should have shape (num_images, 4, 64, 64) in this case
    return latents


def generate_images_with_depth_prior(args, pipe, init_images_folder_name, mask_folder_name, 
                                    smpl_depth_folder_name, out_folder_name, save_grid=False):
    logger.info("Generating images with depth prior...")
    init_image_path = os.path.join(args.out_path, init_images_folder_name)
    init_mask_path = os.path.join(args.out_path, mask_folder_name)
    smpl_depth_path = os.path.join(args.out_path, smpl_depth_folder_name)
    out_path = os.path.join(args.out_path, out_folder_name)
    img_size = 512
    depth_size = 64

    files = os.listdir(smpl_depth_path)
    for hmr_depth_file_name in files:
        file_name, file_ext = os.path.splitext(hmr_depth_file_name)
        if file_name.endswith('_grid'): continue

        file_parts = file_name.split('_')
        file_name = '_'.join(file_parts[:-1])
        sample_idx = int(file_parts[-1])
        original_file_name = file_name + file_ext


        if os.path.exists(os.path.join(out_path, f'{file_name}_{sample_idx}.png')):
            continue

        depth_path = os.path.join(smpl_depth_path, hmr_depth_file_name)
        image_path = os.path.join(init_image_path, original_file_name)
        mask_path = os.path.join(init_mask_path, original_file_name)

        try:
            depth_map = PIL.Image.open(depth_path).convert('L')
            depth_map_small = depth_map.resize((depth_size, depth_size))
        except:
            logger.info(f"{depth_path} does not exist.")
            continue
        try:
            mask = PIL.Image.open(mask_path).convert('L').resize((depth_size, depth_size))
        except:
            logger.info(f"{mask_path} does not exist.")
            continue
        depth_map = torch.from_numpy(np.array(depth_map)).to(device=pipe.device, dtype=torch_dtype).unsqueeze(0)
        # depth_map *= 2
        image = PIL.Image.open(image_path).convert('RGB')
        image = image.resize((img_size, img_size))
        mask = np.array(mask) / 255.

        # merge with depth map.
        depth_map = (depth_map.max() + 50 - depth_map ) * (depth_map > 0).float()
        foreground = (np.array(depth_map_small) > 0).astype(np.float32)
        mask = ((mask + foreground) > 0).astype(np.float32)
        mask = torch.from_numpy(mask).to(device=pipe.device, dtype=torch_dtype).unsqueeze(0).unsqueeze(0)
        
        generator = torch.Generator(device=device)
        with torch.autocast(device):
            # n_prompt = "bad, deformed, ugly, bad anatomy"  # from the depth2image tutorial.
            n_prompt = "deformed, bad anatomy"
            _, prompt = sample_prompt(args.action)
            random_latents = get_random_latents(generator, device)
            image_in = prepare_image(image).to(device=device, dtype=torch_dtype)  # [1, 3, 512, 512]
            image_latents = pipe.vae.encode(image_in).latent_dist.sample(generator=generator)
            image_latents *= pipe.scheduler.init_noise_sigma * 0.18215  # magic number

            if not args.use_random_latents_for_human:
                combined_latents = image_latents
            else:
                combined_latents = image_latents * (1 - mask) + random_latents * mask
            images, _ = pipe(
                prompt=prompt,
                image=image,
                latents=combined_latents,
                guidance_scale = 7.5,
                num_inference_steps = args.num_inference_steps,
                depth_map = depth_map,
                negative_prompt=n_prompt,
                strength=args.strength,
                generator=generator
            )
        out_image = images.images[0]
        if args.add_blur:
            blur_radius = np.random.uniform(1., 2.5)
            out_image = out_image.filter(PIL.ImageFilter.GaussianBlur(radius=blur_radius))
            
        if save_grid:
            out_grid_image = image_grid([
                image, PIL.Image.open(depth_path).convert('L').resize((img_size, img_size)), out_image
            ], 1, 3)
            out_grid_image.save(os.path.join(out_path, f'{file_name}_{sample_idx}_grid.png'))
        
        out_image.save(os.path.join(out_path, f'{file_name}_{sample_idx}.png'))
        # PIL.Image.fromarray(depth_array).convert('L').save(os.path.join(out_path, f'depth_{file_name}'))
    return


def get_depth_pipeline(device):
    torch_dtype = torch.float16
    depth_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        revision="fp16",
        torch_dtype=torch_dtype,
        use_auth_token=False
    ).to(device)
    return depth_pipe


def get_pipeline(device):
    torch_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4',
        revision="fp16",
        torch_dtype=torch_dtype,
        use_auth_token=False
    ).to(device)
    return pipe


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    device = "cuda"
    for subfoler in ['init_images', 'masks', 'final_images']:
        os.makedirs(os.path.join(args.out_path, subfoler), exist_ok=True)

    torch_dtype = torch.float16   # this could fit in a 12 GB GPU.
    
    if not args.use_real_guide:
        pipe = get_pipeline(device)
        # 1. Generate images without depth prior
        generate_images(args, pipe, 'init_images')

    else:
        from downstream_utils.prepare_real_guidance import (
            write_ski_images, write_gymnastics, write_polevault_or_highjump,
            write_diving
            )
        # 1. Write real image as guidance to generate synthetic images.
        if args.real_guidance_action == 'ski':
            write_ski_images(args.out_path, args.rg_num_samples)
        elif args.real_guidance_action in ['polevault', 'highjump']:
            write_polevault_or_highjump(args.out_path, action=args.real_guidance_action)
        elif args.real_guidance_action in ['balancebeam', 'vault', 'unevenbars']:
            write_gymnastics(args.out_path, action=args.real_guidance_action)
        elif args.real_guidance_action == 'diving':
            write_diving(args.out_path)
        else:
            # TODO: implement other gymnastics actions
            raise NotImplementedError(f'Action {args.real_guidance_action} not implemented')


    # 2. Generate masks and depth maps
    run_maskrcnn(args, 'init_images', 'masks')
    torch.cuda.empty_cache()

    pipe = get_depth_pipeline(device)

    # 3. Generate images with depth prior
    if args.hmr_method == 'spin':
        dapa_path = './external/DAPA_release'
        sys.path.append(dapa_path)
        from hmr_utils import SPIN_wrapper
        out_folder_name = 'spin'
        os.makedirs(os.path.join(args.out_path, out_folder_name), exist_ok=True)
        runner = SPIN_wrapper('external/DAPA_release/data/model_checkpoint.pt', 'cuda')

        annotations = runner.inference(
            args, 'init_images', out_folder_name, apply_pose_aug=args.apply_pose_aug, num_aug_samples=args.k, 
            save_grid=False, move_person_to_center=not args.use_real_guide)
        with open(os.path.join(args.out_path, 'spin_gt.pkl'), 'wb+') as f:
            pickle.dump(annotations, f)

    elif args.hmr_method == 'bev':
        from bev_utils import run_bev
        out_folder_name = 'bev'
        os.makedirs(os.path.join(args.out_path, out_folder_name), exist_ok=True)
        os.makedirs(os.path.join(args.out_path, f'{out_folder_name}_npz'), exist_ok=True)
        run_bev(args, 'init_images', out_folder_name, apply_pose_aug=args.apply_pose_aug)

    generate_images_with_depth_prior(
        args, pipe, 'init_images', 'masks', out_folder_name, 'final_images', save_grid=args.num_opt_cycles < 1)

    if args.hmr_method == 'bev':
        sys.exit(0)  # Done here. No need to run the rest of the code.

    # runner.finetune('spin_gt.pkl', generation_path=args.out_path, image_folder='final_images', num_epochs=20)

    for iter in range(1, args.num_opt_cycles+1):
        for subfoler in [f'init_images_iter{iter}', f'masks_iter{iter}',
                        f'{out_folder_name}_iter{iter}', f'final_images_iter{iter}']:
            os.makedirs(os.path.join(args.out_path, subfoler), exist_ok=True)
        
        save_grid = iter == args.num_opt_cycles

        hmr_folder = f'{out_folder_name}_iter{iter}'
        final_images_folder = f'final_images_iter{iter}'

        if not args.use_real_guide:
            init_images_folder = f'init_images_iter{iter}'
            masks_folder = f'masks_iter{iter}'
            pipe = get_pipeline()
            generate_images(args, pipe, init_images_folder)
            run_maskrcnn(args, init_images_folder, masks_folder)
            torch.cuda.empty_cache()
        else:
            init_images_folder = 'init_images'
            masks_folder = 'masks'

        if args.hmr_method == 'spin':
            annotations = runner.inference(
                args, init_images_folder, hmr_folder, apply_pose_aug=args.apply_pose_aug, 
                num_aug_samples=args.k, save_grid=save_grid, move_person_to_center=False)

            with open(os.path.join(args.out_path, f'spin_gt_iter{iter}.pkl'), 'wb+') as f:
                pickle.dump(annotations, f)

        elif args.hmr_method == 'bev':
            run_bev(args, init_images_folder, hmr_folder, apply_pose_aug=args.apply_pose_aug)
        else:
            raise NotImplementedError(f'HMR method {args.hmr_method} not implemented')
        
        pipe = get_depth_pipeline()
        generate_images_with_depth_prior(
            args, pipe, init_images_folder, masks_folder, 
            hmr_folder, final_images_folder, save_grid)

        if iter < args.num_opt_cycles:
            runner.finetune(
                annot_name=f'spin_gt_iter{iter}.pkl', generation_path=args.out_path, 
                image_folder=final_images_folder, num_epochs=20)
