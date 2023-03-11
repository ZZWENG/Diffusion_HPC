"""Dump images from the dataset for evaluation.
"""
import os
import torch
from torchvision import transforms as T
from tqdm import tqdm

from datasets import BaseDataset

device = 'cpu'

def dump_images(dataset, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        images = item['img'].to(device).unsqueeze(0)
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        # import pdb; pdb.set_trace()
        T.ToPILImage()(images[0]).save(os.path.join(out_path, f'image_{i}.png'))


if __name__ == '__main__':

    for dataset_name in [
            'highjump_real', 'balancebeam_real', 'diving_real', 
            'polevault_real', 'vault_real', 'unevenbars_real'
        ]:
        out = '/oak/stanford/groups/syyeung/zzweng/hmr_eval'
    
        dataset = BaseDataset(None, dataset_name, is_train=False)
        dump_images(dataset, os.path.join(out, dataset_name))

        keypoints = torch.stack([dataset[i]['keypoints'] for i in range(len(dataset))])
        torch.save(keypoints, os.path.join(out, f'{dataset_name}_keypoints.pt'))
