import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from vis_utils import overlay_kp


base_path = r'./data/SportsCap_Dataset_SMART_v1'
files = [
    'Table_VideoInfo_diving.json', 
    'Table_VideoInfo_polevalut_highjump_badminton.json', 
    'Table_VideoInfo_gym.json']

out_size = 512  # SD takes (512, 512) images as input
person_dim = 400  # make the person roughly 200 pixels tall


def _get_padded_bbox(openpose, bbox=None):
    if bbox is None:
        bbox = [min(openpose[:,0]), min(openpose[:,1]),
                max(openpose[:,0]), max(openpose[:,1])]
    box_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    # make it a square
    longer_side = max(box_size)
    bbox = [bbox[0] - (longer_side - box_size[0])/2, bbox[1] - (longer_side - box_size[1])/2,
            bbox[2] + (longer_side - box_size[0])/2, bbox[3] + (longer_side - box_size[1])/2]
    box_size = (longer_side, longer_side)

    # scale the box to make the person roughly person_dim pixels tall, and centered
    scale_factor = person_dim / max(box_size)
    if scale_factor < 1:
        padded_bbox = bbox
    else:
        pad_size = [(out_size - box_size[0])/2/scale_factor, (out_size - box_size[1])/2/scale_factor]
        padded_bbox = [bbox[0]-pad_size[0], bbox[1]-pad_size[1],
                    bbox[2]+pad_size[0], bbox[3]+pad_size[1]]

    openpose[:,0] = (openpose[:,0] - padded_bbox[0]) * out_size / (padded_bbox[2] - padded_bbox[0])
    openpose[:,1] = (openpose[:,1] - padded_bbox[1]) * out_size / (padded_bbox[3] - padded_bbox[1])
    return openpose, padded_bbox


def write_polevault_or_highjump(out_path, action="polevault", debug=False):
    file = files[1]
    if action == "polevault":
        action_label = 'cgt'
    elif action == "highjump":
        action_label = 'tg'
    else:
        raise NotImplementedError()

    annot_path = os.path.join(base_path, 'annotations', file)
    with open(annot_path) as f:
        data = json.load(f)

    valid_videos = [vid['action_labels'] for vid in data if action_label in vid['action_labels']]
    train_len = 2
    print(f"Total {len(valid_videos)} videos. Using {train_len} for training.")
    valid_labels = valid_videos[:train_len]

    for vid in tqdm(data):
        if vid['action_labels'] not in valid_labels: continue

        for meta in vid['frames']:
            if len(meta['joints']) == 0: continue

            img_path = os.path.join(base_path, 'images', meta['img_name'])
            out_img_path = os.path.join(out_path, 'init_images', meta['img_name'])
            openpose = np.array(meta['joints'])
            openpose, padded_bbox = _get_padded_bbox(openpose)
            meta['joints'] = openpose.tolist()

            image = Image.open(img_path).crop(padded_bbox).resize((out_size, out_size))
            
            if debug:
                image = overlay_kp(image, openpose)
                image.save('debug.png')
                import pdb; pdb.set_trace()
            
            image.save(out_img_path.replace('.jpg', '.png'))

    # with open(os.path.join(out_path, 'annotations.json'), 'w') as f:
        # json.dump(data, f)


def write_gymnastics(out_path, action='balancebeam', debug=False):
    if action == 'balancebeam':
        action_label = '184;204'  # 20 videos, 444 frames. 
        train_len = 5
    elif action == 'vault':
        action_label = 'vt'  # 11 videos, 242 frames.
        train_len = 4
    elif action == 'unevenbars':
        action_label = '271'
        train_len = 3
    else:
        # TODO: add other actions
        raise NotImplementedError()

    file = files[2]

    annot_path = os.path.join(base_path, 'annotations', file)
    with open(annot_path) as f:
        data = json.load(f)

    valid_actions = [vid['action_labels'] for vid in data]
    print(set(valid_actions))
    valid_videos = [vid['VideoName'] for vid in data if vid['action_labels'] in action_label]

    print(f"Total {len(valid_videos)} videos. Using {train_len} for training.")
    valid_labels = valid_videos[:train_len]

    num_frames = 0
    for vid in tqdm(data):
        if vid['VideoName'] not in valid_labels: continue

        for meta in vid['frames']:
            if len(meta['joints']) == 0: continue

            img_path = os.path.join(base_path, 'images', meta['img_name'])
            out_img_path = os.path.join(out_path, 'init_images', meta['img_name'])
            openpose = np.array(meta['joints'])
            openpose, padded_bbox = _get_padded_bbox(openpose)
            meta['joints'] = openpose.tolist()
            image = Image.open(img_path).crop(padded_bbox).resize((out_size, out_size))

            if debug:
                image = overlay_kp(image, openpose)
                image.save('debug.png')
                import pdb; pdb.set_trace()
            
            image.save(out_img_path.replace('.jpg', '.png'))
            num_frames += 1

    print(f"Saved total {num_frames} frames to {os.path.join(out_path, 'init_images')}.")
    # with open(os.path.join(out_path, 'annotations.json'), 'w') as f:
        # json.dump(data, f)


def write_diving(out_path, debug=False):
    file = files[0]

    annot_path = os.path.join(base_path, 'annotations', file)
    with open(annot_path) as f:
        data = json.load(f)

    train_len = 15  # this is roughly 8000 frames (10% of total valid frames)
    print(f"Total {len(data)} videos. Using {train_len} for training.")
    data = data[:train_len]

    num_frames = 0
    for vid in tqdm(data):
        for meta in vid['frames'][::10]:
            if len(meta['joints']) == 0: continue

            img_path = os.path.join(base_path, 'images', meta['img_name'])
            out_img_path = os.path.join(out_path, 'init_images', meta['img_name'])
            bbox = np.array(meta['bbox'])
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            openpose = np.array(meta['joints'])
            openpose, padded_bbox = _get_padded_bbox(openpose, bbox)
            meta['joints'] = openpose.tolist()
            image = Image.open(img_path).crop(padded_bbox).resize((out_size, out_size))

            if debug:
                image = overlay_kp(image, openpose)
                image.save('debug.png')
                import pdb; pdb.set_trace()
            
            image.save(out_img_path.replace('.jpg', '.png'))
            num_frames += 1

    print(f"Saved total {num_frames} frames to {os.path.join(out_path, 'init_images')}.")



if __name__ == '__main__':
    out_path = './diving_downsampled_rg'
    os.makedirs(os.path.join(out_path, 'init_images'), exist_ok=True)
    write_diving(out_path, debug=False)
