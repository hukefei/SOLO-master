import os, shutil
import json
import skimage.io as io
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def showAnns(json_file, img_dir, mask_dir):
    coco = COCO(json_file)
    with open(json_file,'r') as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    for img in images:
        img_id = img['id']
        I = io.imread(os.path.join(img_dir, img['file_name']))
        plt.figure()
        plt.clf()
        plt.axis('off')
        plt.imshow(I)
        anns = []
        for ann in annotations:
            if ann['image_id'] == img_id:
                anns.append(ann)
        coco.showAnns(anns)
        plt.savefig(os.path.join(mask_dir, img['file_name']), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    json_file = '/Users/dyy/Desktop/datasets/seg/train.json'
    img_dir = '/Users/dyy/Desktop/datasets/seg/JPEGimages'
    mask_dir = '/Users/dyy/Desktop/datasets/seg/mask'
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir)
    showAnns(json_file, img_dir, mask_dir)

