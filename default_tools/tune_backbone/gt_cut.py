#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import json


def get_defect_patch(json_file, img_dir, save_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    for i, img in enumerate(images):
        img_name = img['file_name']
        print(i, img_name)
        img_id = img['id']
        image = cv2.imread(os.path.join(img_dir, img_name))
        for j, anno in enumerate(annotations):
            if anno['image_id'] == img_id:
                category = anno['category_id']
                bbox = anno['bbox']
                x, y, w, h = [round(c) for c in bbox]
                if max(w, h) < 30:
                    continue
                patch = image[y:(y+h), x:(x+w), :]

                save_path = os.path.join(save_dir, str(category))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(os.path.join(save_path, img_name[:-4]+'_'+str(j)+'.jpg'), patch)


if __name__ == '__main__':
    json_file = '/home/versa/dataset/MSCOCO/train_aug.json'
    img_dir = '/home/versa/dataset/MSCOCO/train2017/'
    save_dir = '/home/versa/dataset/MSCOCO/train_gt'
    get_defect_patch(json_file, img_dir, save_dir)