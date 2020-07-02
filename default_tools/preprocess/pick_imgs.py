import json
import os, shutil
from pycocotools.coco import COCO

dataset = 'val'
img_dir = '/versa/dataset/COCO2017/coco/{}2017'.format(dataset)
save_dir = '/versa/dyy/coco/{}'.format(dataset)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)
annFile = '/versa/dataset/COCO2017/coco/annotations/instances_{}2017.json'.format(dataset)
coco = COCO(annFile)

# cat_ids = coco.getCatIds()
# print(len(cat_ids))
# img_ids = coco.getImgIds()
# print(len(img_ids))
#
# cats = coco.loadCats(cat_ids)
# cat_names = [cat['name'] for cat in cats]
# super_names = set([cat['supercategory'] for cat in cats])
# print(cat_names)
# print(super_names)

catIds = coco.getCatIds(catNms=['person', 'cat', 'dog', 'teddy bear'])
categories = coco.loadCats(catIds)
print(categories)

img_ids = []
annotations = []
for cat_id in catIds:
    imgIds = coco.getImgIds(catIds=[cat_id])
    img_ids.extend(imgIds)
    for img_id in imgIds:
        annIds = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if ann['category_id'] in catIds:
                annotations.append(ann)

img_ids = list(set(img_ids))
images = []
for img_id in img_ids:
    info = coco.loadImgs([img_id])[0]
    images.append(info)
    img_name = info['file_name']
    shutil.copy(os.path.join(img_dir, img_name),
                os.path.join(save_dir, img_name))
print(len(images))

json_dict = {"images": images, "annotations": annotations, "categories": categories}
with open('/versa/dyy/coco/{}.json'.format(dataset), 'w') as f:
    json.dump(json_dict, f)
