import os, glob, shutil
import numpy as np

person_dir = '/versa/dyy/coco/val_gt/1'
imgs = glob.glob(os.path.join(person_dir, '*.jpg'))
print(len(imgs))
imgs = np.random.choice(imgs, 500)
print(len(imgs))
save_dir = '/versa/dyy/coco/val_gt/1_'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for img in imgs:
    name = img.split('/')[-1]
    try:
        shutil.move(img, os.path.join(save_dir, name))
    except:
        continue