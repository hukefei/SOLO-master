
import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils


annFile = '/versa/dyy/coco/val.json'
coco = COCO(annFile)
img_info = coco.loadImgs([137294])[0]
print(img_info)
ann_ids = coco.getAnnIds(imgIds=[137294], catIds=[1])
anns = coco.loadAnns(ann_ids)
print(anns)
mask_anns = [ann['segmentation'] for ann in anns]
img_h = img_info['height']
img_w = img_info['width']
rles = maskUtils.frPyObjects(mask_anns[1], img_h, img_w)
rle = maskUtils.merge(rles)
mask = maskUtils.decode(rle)
# cv2.imwrite('mask.png', mask*255)

# mask = cv2.imread('mask.png')
# imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(mask*255, 127, 255, 0)
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt_mask = np.zeros((img_h, img_w, 3))
cnt_mask = cv2.drawContours(cnt_mask, cnts, -1, (255, 255, 255), 1).astype('uint8')
cnt_mask = cv2.cvtColor(cnt_mask, cv2.COLOR_BGR2GRAY)
cnt_mask = (cnt_mask / 255).astype('uint8')
print(cnt_mask)
# cv2.imwrite('contour.png', cnt_mask)