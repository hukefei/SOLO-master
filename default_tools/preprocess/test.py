import cv2
import numpy as np
import os, glob, shutil
import json
import re
import fnmatch
from PIL import Image
from pycococreatortools import pycococreatortools


def rgb2masks(label_img, save_dir):
    lbl_name = os.path.split(label_img)[-1].split('.')[0]
    lbl = cv2.imread(label_img, 1)
    h, w = lbl.shape[:2]
    lbl_dict = {}
    idx = 0
    white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(h):
        for j in range(w):
            if tuple(lbl[i][j]) in lbl_dict or \
                    tuple(lbl[i][j]) == (0, 0, 0) or \
                    tuple(lbl[i][j]) == (255, 255, 255):
                continue
            lbl_dict[tuple(lbl[i][j])] = idx
            mask = (lbl == lbl[i][j]).all(-1)
            binary_mask = np.where(mask[..., None], white_mask, 0)
            mask_name = lbl_name + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(save_dir, mask_name), binary_mask)
            idx += 1
    # print(lbl_dict)


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files


if __name__ == '__main__':
    root_dir = '/Users/dyy/Desktop/human_seg_datasets'
    img_dir = '/Users/dyy/Desktop/human/images'
    lbl_dir = '/Users/dyy/Desktop/human/labels'
    xml_dir = '/Users/dyy/Desktop/human/xmls'
    for dir in [img_dir, lbl_dir, xml_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            if root.split('/')[-1].startswith('2019'):
                if dir == 'segmentationObject':
                    labels = glob.glob(os.path.join(root, dir, '*.png'))
                    for label in labels:
                        name = label.split('/')[-3][:12] + '_' + label.split('/')[-1]
                        shutil.copy(label, os.path.join(lbl_dir, name))

                if dir == 'JPEGimages':
                    imgs = glob.glob(os.path.join(root, dir, '*.jpg'))
                    for img in imgs:
                        name = img.split('/')[-3][:12] + '_' + img.split('/')[-1]
                        shutil.copy(img, os.path.join(img_dir, name))

                if dir == 'VOC':
                    xmls = glob.glob(os.path.join(root, dir, '*.xml'))
                    for xml in xmls:
                        name = xml.split('/')[-3][:12] + '_' + xml.split('/')[-1]
                        shutil.copy(xml, os.path.join(xml_dir, name))

    label_dir = '/Users/dyy/Desktop/human/labels/'
    save_dir = '/Users/dyy/Desktop/human/masks/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_list = glob.glob(os.path.join(label_dir, '*.png'))[:1]
    for i, label_name in enumerate(label_list):
        print(i, label_name)
        rgb2masks(label_name, save_dir)

    ROOT_DIR = '/Users/dyy/Desktop/human'
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "masks")

    CATEGORIES = [
        {
            'supercategory': 'none',
            'name': '0',
            'id': 0
        },
        {
            'supercategory': 'none',
            'name': '1',
            'id': 1
        },
        {
            'supercategory': 'none',
            'name': '2',
            'id': 2
        }
    ]

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/test.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

