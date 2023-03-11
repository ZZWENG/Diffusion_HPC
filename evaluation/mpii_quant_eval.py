import argparse
import csv
import os
from shutil import copy

import numpy as np
import scipy.io as sio
import torch

from api import our_pipeline, get_pipeline


# read annotations
def generate_dataset_obj(obj):
    must_be_list_fields = ["annolist", "annorect", "point", "img_train", "single_person", "act", "video_list"]
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))

    elif type(obj) == sio.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields and type(field) != list:
                field = [field]
            ret[field_name] = field

    else:
        ret = obj

    return ret


def save_csv(my_dict, save_path, col_names):
    with open(save_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(my_dict)

prefixes = [
    'a photo of',
    'a nice photo of',
    'a high resolution photo of',
    'a pretty picture of',
    'an eye level shot of',
    'a hip level shot of',
    'a full body photo of',
    'a full body shot of',
    'a full shot of',
    'a photo of',
]


def get_text_prompt(cat):
    subject_list = [
        'a person',
        'a woman',
        'women',
        'men',
        'a man',
        'a child',
        'children',
        'a old person',
        'a teenager',
        'a young person',
        'several people',
        'multiple people',
        'some people',
        'a group of people',
    ]
    prefix_list = [ 
        'a photo of',
        'a nice photo of',
        'a high resolution photo of',
        'a pretty picture of',
        'an eye level shot of',
        'a full body photo of',
        'a full body shot of',
        'a full shot of',
        'a pretty photo of'
    ]

    prefix = np.random.choice(prefix_list)
    subject = np.random.choice(subject_list)
    text_prompt = f'{prefix} {subject} doing {cat}'
    return text_prompt


def sample_images(args, cat_info_map, out_folder, num_cat_ims=100):
    os.makedirs(os.path.join(out_folder, 'ours'), exist_ok=True)
    os.makedirs(os.path.join(out_folder, 'stable_diffusion'), exist_ok=True)

    np.random.seed(args.part_id)
    torch.manual_seed(args.part_id)

    pipe = get_pipeline('cuda', finetuned_on=args.finetuned_on)

    print(f'Sampling {num_cat_ims} images per category')
    sample_info = [] # annotator GT csv
    for cat, ims_info in cat_info_map.items():
        if cat in ['miscellaneous', 'inactivity quiet/light']: continue

        choices = np.random.choice(ims_info, num_cat_ims)
        for idx, img_info in enumerate(choices):
            text_prompt = get_text_prompt(cat)

            # generate synthetic images
            outputs = our_pipeline(pipe, text_prompt, real_image=None, hmr_method='bev',
                                   filter_by_vposer=True)
            if outputs is None: continue
            sd_image, our_image = outputs

            our_image = our_image.resize((299, 299))  # default size used by clean-fid
            sd_image = sd_image.resize((299, 299))

            img_name = f'{args.part_id}_{idx}_' + img_info['img_name']
            save_path = os.path.join(out_folder, 'quant_text_eval_mpii', 'ours', img_name)
            our_image.save(save_path)
            save_path = os.path.join(out_folder, 'quant_text_eval_mpii', 'stable_diffusion', img_name) 
            sd_image.save(save_path)
            sample_info.append(
                {'img_name': img_name, 'cat': cat, 'act': img_info['act'], 'act_id': img_info['act_id'],
                 'text_prompt': text_prompt})

        save_csv(sample_info, os.path.join(out_folder, f'{args.part_id}_all_info.csv'), list(sample_info[0].keys()))
    save_csv(sample_info, os.path.join(out_folder, f'{args.part_id}_all_info.csv'), list(sample_info[0].keys()))


def main(args):
    # read MPII annotations and split images per category
    mpii_anns_path = os.path.join(args.mpii_base_path, r'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
    decoded1 = sio.loadmat(mpii_anns_path, struct_as_record=False)["RELEASE"]
    obj = generate_dataset_obj(decoded1)
    eft_data = np.load(os.path.join(args.mpii_base_path, 'eft_annots.npz'), allow_pickle=True)['annots'][()]

    cat_names = [v['cat_name'] for v in obj['act'] if len(v['cat_name']) > 0]
    categories = np.unique(cat_names)
    num_cat_ims = int(np.round(args.num_images/len(categories)))

    cat_info_map = {}
    for idx, act_info in enumerate(obj['act']):
        this_cat = act_info['cat_name']
        img_name = obj['annolist'][idx]['image']['name']
        if len(this_cat) > 0 and img_name in eft_data:
            if this_cat not in cat_info_map.keys():
                cat_info_map[this_cat] = []
            cat_info_map[this_cat].append({'idx': idx, 'act': act_info['act_name'], 'img_name': img_name, 'act_id': act_info['act_id']})

    # run sample_images for obtaining the eval dataset
    sample_images(args, cat_info_map, args.save_folder, num_cat_ims=num_cat_ims)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpii_base_path', type=str, default='./data/mpii')
    parser.add_argument('--save_folder', type=str, default='./eval_data')
    parser.add_argument('--finetuned_on', type=str, default=None, choices=['mpii', None], help='Dataset that SD was finetuned on.')
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--part_id', type=int, default=0, help='part id for parallelization')
    args = parser.parse_args()

    main(args)
