#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, shutil
import json
import random
import getArea
import xml.etree.ElementTree as ET


START_IMAGE_ID = 1
START_BOUNDING_BOX_ID = 1
CATEGORIES = {'背景': 0, '人物': 1, "狗": 3}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' %(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' %(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xmls):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    bnd_id = START_BOUNDING_BOX_ID
    image_id = START_IMAGE_ID
    # categories = ['人物', '背景', '天空', '轿车', '其他', '巴士', '狗', '马']
    for i, xml in enumerate(xmls):
        print(i, xml)
        name = xml.split('/')[-1].replace('.xml', '.jpg')
        tree = ET.parse(xml)
        root = tree.getroot()

        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': name, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            # if category not in categories:
            #     categories.append(category)
            try:
                category_id = CATEGORIES[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                assert len(bndbox) % 2 == 0
                points = []
                for i in range(len(bndbox) // 2):
                    points.append(float(get_and_check(bndbox, 'x{}'.format(i+1), 1).text))
                    points.append(float(get_and_check(bndbox, 'y{}'.format(i+1), 1).text))
                all_x, all_y = points[0::2], points[1::2]
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)
                o_width = xmax - xmin
                o_height = ymax - ymin
                area = getArea.GetAreaOfPolyGon(all_x, all_y)
                ann = {'image_id': image_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'area': area, 'iscrowd': 0, 'ignore': 0,
                       'category_id': category_id, 'id': bnd_id, 'segmentation': [points]}
                json_dict['annotations'].append(ann)
                bnd_id += 1
            except:
                continue
        image_id += 1

    categories = [
        {'supercategory': 'background', 'id': 0, 'name': 'background'},
        {'supercategory': 'person', 'id': 1, 'name': 'person'},
        {'supercategory': 'dog', 'id': 3, 'name': 'dog'},
    ]
    json_dict['catgories'] = categories

    return json_dict


if __name__ == '__main__':
    # root_dir = '/Users/dyy/Desktop/human_seg_datasets'
    # img_dir = '/Users/dyy/Desktop/human/images'
    # lbl_dir = '/Users/dyy/Desktop/human/labels'
    # xml_dir = '/Users/dyy/Desktop/human/xmls'
    # for dir in [img_dir, lbl_dir, xml_dir]:
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #
    # for root, dirs, files in os.walk(root_dir):
    #     for dir in dirs:
    #         if root.split('/')[-1].startswith('2019'):
    #             if dir == 'segmentationObject':
    #                 labels = glob.glob(os.path.join(root, dir, '*.png'))
    #                 for label in labels:
    #                     name = label.split('/')[-3][:12] + '_' + label.split('/')[-1]
    #                     shutil.copy(label, os.path.join(lbl_dir, name))
    #
    #             if dir == 'JPEGimages':
    #                 imgs = glob.glob(os.path.join(root, dir, '*.jpg'))
    #                 for img in imgs:
    #                     name = img.split('/')[-3][:12] + '_' + img.split('/')[-1]
    #                     shutil.copy(img, os.path.join(img_dir, name))
    #
    #             if dir == 'VOC':
    #                 xmls = glob.glob(os.path.join(root, dir, '*.xml'))
    #                 for xml in xmls:
    #                     name = xml.split('/')[-3][:12] + '_' + xml.split('/')[-1]
    #                     shutil.copy(xml, os.path.join(xml_dir, name))

    xmls = glob.glob('/Users/dyy/Desktop/datasets/human/xmls/*.xml')
    train = random.sample(xmls, int(len(xmls)*0.8))
    val = list(set(xmls) - set(train))

    img_dir = '/Users/dyy/Desktop/datasets/human/images'
    train_dir = '/Users/dyy/Desktop/datasets/human/train'
    val_dir = '/Users/dyy/Desktop/datasets/human/val'
    for dir in [train_dir, val_dir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    for xml in train:
        name = xml.split('/')[-1].replace('.xml', '.jpg')
        src = os.path.join(img_dir, name)
        dst = os.path.join(train_dir, name)
        shutil.copy(src, dst)

    for xml in val:
        name = xml.split('/')[-1].replace('.xml', '.jpg')
        src = os.path.join(img_dir, name)
        dst = os.path.join(val_dir, name)
        shutil.copy(src, dst)

    train_dict = convert(train)
    val_dict =convert(val)

    json_fp = open('/Users/dyy/Desktop/datasets/human/train.json', 'w')
    json_str = json.dumps(train_dict)
    json_fp.write(json_str)
    json_fp.close()

    json_fp = open('/Users/dyy/Desktop/datasets/human/val.json', 'w')
    json_str = json.dumps(val_dict)
    json_fp.write(json_str)
    json_fp.close()
