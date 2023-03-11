"""Produce human masks and save them to disk. """
import os, sys
import PIL
import numpy as np
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def get_mask_rcnn_predictor():
    """Get a mask rcnn predictor. """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


# mean accumulator
class MeanAccumulator:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add(self, x):
        self.sum += x
        self.count += 1

    def mean(self):
        return self.sum / self.count


def save_human_masks_single_person(image, acc, predictor, out_folder, file):
    """Get human masks from an image. """
    image_size = image.size[0]
    predictor = get_mask_rcnn_predictor()

    image_np = np.array(image)
    outputs = predictor(image_np)

    is_person = (outputs["instances"].pred_classes == 0).cpu().numpy()
    is_person = np.where(is_person)[0]
    mask = np.zeros((image_size, image_size))
    for idx in is_person[:1]:
        mask = outputs["instances"].pred_masks[idx].cpu().numpy()
        masked_image = PIL.Image.fromarray(image_np * (mask > 0)[:,:,np.newaxis])
        box = outputs["instances"].pred_boxes.tensor[idx].cpu().numpy()
        box = box.astype(int) # (left, upper, right, lower)
        center_x = (box[0] + box[2]) // 2
        center_y = (box[1] + box[3]) // 2
        box_size = max(box[2] - box[0], box[3] - box[1])
        box = (center_x - box_size // 2, center_y - box_size // 2, center_x + box_size // 2, center_y + box_size // 2)

        masked_image = masked_image.crop(box)
        masked_image.save(os.path.join(out_folder, os.path.split(file)[1].replace('.jpg', f'_{idx}.jpg')))

    person_scores = outputs["instances"].scores[outputs["instances"].pred_classes==0].cpu().numpy()
    for score in person_scores:
        acc.add(score)
    return 



def save_human_masks(image, acc, predictor, out_folder, file):
    """Get human masks from an image. """
    image_size = image.size[0]
    predictor = get_mask_rcnn_predictor()

    image_np = np.array(image)
    outputs = predictor(image_np)

    is_person = (outputs["instances"].pred_classes == 0).cpu().numpy()
    is_person = np.where(is_person)[0]
    mask = np.zeros((image_size, image_size))
    for idx in is_person:
        mask += outputs["instances"].pred_masks[idx].cpu().numpy()

    masked_image = PIL.Image.fromarray(image_np * (mask > 0)[:,:,np.newaxis])    
    masked_image.save(os.path.join(out_folder, os.path.split(file)[1]))

    person_scores = outputs["instances"].scores[outputs["instances"].pred_classes==0].cpu().numpy()
    for score in person_scores:
        acc.add(score)
    return 


def process_folder(folder, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    is_single_person = 'smart' in folder
    files = os.listdir(folder)
    predictor = get_mask_rcnn_predictor()
    acc = MeanAccumulator()
    for file in tqdm(files, desc='Processing images'):
        if file.startswith('.') or file.endswith('jsonl'): continue
        image = PIL.Image.open(os.path.join(folder, file)).resize((299, 299))
        if is_single_person:
            save_human_masks_single_person(image, acc, predictor, out_folder, file)
        else:
            save_human_masks(image, acc, predictor, out_folder, file)

    print(f'Average person score: {acc.mean():.3f}')
    print(f'Number of persons: {acc.count}')

if __name__ == '__main__':
    path = sys.argv[1]
    if path.endswith('/'): path = path[:-1]
    
    print(path)
    process_folder(
        path,
        path + '_masked'
    )
