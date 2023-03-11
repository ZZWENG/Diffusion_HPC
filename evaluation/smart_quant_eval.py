import csv
import os
import torch

import numpy as np
import argparse

from api import our_pipeline, get_pipeline


def save_csv(my_dict, save_path, col_names):
    with open(save_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(my_dict)


def get_text_prompt(action_name):
    if action_name == 'balancebeam':
        action_str = "an athlete is doing gymnastics on balance beam"
    elif action_name == 'highjump':
        action_str = "an athlete is doing high jump"
    elif action_name == 'diving':
        action_str = "an athlete is diving"
    elif action_name == 'polevault':
        action_str = "an athlete is doing pole vault"
    elif action_name == 'vault':
        action_str = "an athlete is doing vault"
    elif action_name == 'unevenbars':
        action_str = "an athlete is doing gymnastics on uneven bars"
    else:
        raise NotImplementedError(f'Action {action_name} not implemented.')
    return action_str


def sample_images(args, out_folder, num_cat_ims=100):
    os.makedirs(os.path.join(out_folder, 'ours'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'stable_diffusion'), exist_ok=True)

    np.random.seed(args.part_id)
    torch.manual_seed(args.part_id)
    pipe = get_pipeline('cuda', finetuned_on=args.finetuned_on)

    print(f'Sampling {num_cat_ims} images per category')
    sample_info = [] # annotator GT csv
    for idx in range(num_cat_ims): 
        action_name = np.random.choice(['balancebeam', 'highjump', 'diving', 'polevault', 'vault', 'unevenbars'])
        text_prompt = get_text_prompt(action_name)
        print(text_prompt)
        # generate synthetic images
        outputs = our_pipeline(pipe, text_prompt, real_image=None, hmr_method='bev', 
                               filter_by_vposer=True, use_random_for_d2i=True)
        if outputs is None: continue
        sd_image, our_image = outputs

        our_image = our_image.resize((299, 299))  # default size used by clean-fid
        sd_image = sd_image.resize((299, 299))

        img_name = f'{args.part_id}_{idx}_{action_name}.png'
        save_path = os.path.join(out_folder, 'quant_text_eval_smart', 'ours', img_name)
        our_image.save(save_path)
        save_path = os.path.join(out_folder, 'quant_text_eval_smart', 'stable_diffusion', img_name) 
        sd_image.save(save_path)
        sample_info.append({'img_name': img_name, 'act': action_name})
    save_csv(sample_info, os.path.join(out_folder, f'{args.part_id}_all_info.csv'), list(sample_info[0].keys()))


def main(args):
    # run sample_images for obtaining the eval dataset
    num_cat_ims = args.num_images // 6
    sample_images(args, args.save_folder, num_cat_ims=num_cat_ims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, default='./eval_data')
    parser.add_argument('--finetuned_on', type=str, default=None, choices=['smart', None], help='Dataset that SD was finetuned on.')
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--part_id', type=int, default=0, help='part id for parallelization')
    args = parser.parse_args()

    main(args)
