import os
from tqdm import tqdm
import h5py
from PIL import Image

data_root = './data/ski_3d'

# load label data
split = 'train'
h5_label_file = h5py.File(os.path.join(data_root, f'./{split}/labels.h5'), 'r')

# train: 8481 examples
# test: 1716 examples

def write_ski_images(out_path, num_samples=200):
    if num_samples == -1:
        # Use all of training set.
        num_samples = len(h5_label_file['seq'])

    for index in tqdm(range(num_samples)):
        seq   = int(h5_label_file['seq'][index])
        cam   = int(h5_label_file['cam'][index])
        frame = int(h5_label_file['frame'][index])
        subj  = int(h5_label_file['subj'][index])
        pose_3D = h5_label_file['3D'][index].reshape([-1,3])
        pose_2D = h5_label_file['2D'][index].reshape([-1,2]) # in range 0..1
        pose_2D_px = 256*pose_2D # in pixels, range 0..255

        # load image
        image_name = os.path.join(data_root, './{}/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(split,seq,cam,frame))
        img = Image.open(image_name)
        img.save(os.path.join(
            out_path, 'init_images', 
            'seq_{:03d}_cam_{:02d}_image_{:06d}.png'.format(seq,cam,frame)
            ))

