import json
import os, shutil
from pycocotools.coco import COCO
import random
random.seed(42)

root = '/Users/dyy/Desktop/datasets/seg'
img_dir = os.path.join(root, 'JPEGimages')
annFile = os.path.join(root, 'dataset.json')
coco = COCO(annFile)

catIds = coco.getCatIds(catNms=['背景', '人物', '猫', '狗', '毛绒玩具'])
categories = coco.loadCats(catIds)
for cate in categories:
    print(cate)

img_ids = []
for cat_id in catIds:
    imgIds = coco.getImgIds(catIds=[cat_id])
    img_ids.extend(imgIds)

train = random.sample(img_ids, int(len(img_ids)*0.8))
val = list(set(img_ids) - set(train))

def get_imgs(imgIds, save_dir):
    images = []
    for img_id in imgIds:
        info = coco.loadImgs([img_id])[0]
        images.append(info)
        img_name = info['file_name']
        shutil.copy(os.path.join(img_dir, img_name),
                    os.path.join(save_dir, img_name))
    return images

def get_anns(imgIds):
    annotations = []
    for img_id in imgIds:
        annIds = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            category_id = ann['category_id']
            if category_id in catIds:
                if category_id == 3:   # background
                    ann['category_id'] = 0
                elif category_id == 1:  # person
                    ann['category_id'] = 1
                elif category_id == 7:  # cat
                    ann['category_id'] = 2
                elif category_id == 8:  # dog
                    ann['category_id'] = 3
                elif category_id in [5, 9, 16, 24, 26]:  # cartoon
                    ann['category_id'] = 4
                annotations.append(ann)
    return annotations

categories = [
    {'supercategory': 'background', 'id': 0, 'name': 'background'},
    {'supercategory': 'person', 'id': 1, 'name': 'person'},
    {'supercategory': 'cat', 'id': 2, 'name': 'cat'},
    {'supercategory': 'dog', 'id': 3, 'name': 'dog'},
    {'supercategory': 'cartoon', 'id': 4, 'name': 'cartoon'},
]

train_dir = os.path.join(root, 'train')
val_dir = os.path.join(root, 'val')
for dir in [train_dir, val_dir]:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

train_json = {"images": get_imgs(train, train_dir), "annotations": get_anns(train), "categories": categories}
with open(os.path.join(root, 'train.json'), 'w') as f:
    json.dump(train_json, f)

val_json = {"images": get_imgs(val, val_dir), "annotations": get_anns(val), "categories": categories}
with open(os.path.join(root, 'val.json'), 'w') as f:
    json.dump(val_json, f)
