from utils import load_config

import os
import cv2
import shutil

import numpy as np
from torchvision.datasets import VOCSegmentation

class_index_to_rgb = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128],
    8: [64, 0, 0],
    9: [192, 0, 0],
    10: [64, 128, 0],
    11: [192, 128, 0],
    12: [64, 0, 128],
    13: [192, 0, 128],
    14: [64, 128, 128],
    15: [192, 128, 128],
    16: [0, 64, 0],
    17: [128, 64, 0],
    18: [0, 192, 0],
    19: [128, 192, 0],
    20: [0, 64, 128]
}

VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0],
                ]

rgb_to_class_index = {tuple(rgb): idx for idx, rgb in class_index_to_rgb.items()}

def dataInit():
    config = load_config.loadConfig()
    print('Data loading it takes few minutes')

    if not os.path.isdir('./data/VOCdevkit/'):
        VOCSegmentation(root='./data', year='2012', image_set='val', download=True)

    if os.path.isdir(f"./data/{config['resolution']}"):
        return

    os.mkdir(f"./data/{config['resolution']}")

    f_names = []

    with open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r") as f:
        for line in f:
            f_names.append(line[:-1])

    images = [cv2.resize(cv2.cvtColor(cv2.imread(f'./data/VOCdevkit/VOC2012/JPEGImages/{f_name}.jpg'),cv2.COLOR_BGR2RGB),(config['resolution'],config['resolution']),interpolation=cv2.INTER_LINEAR) for f_name in f_names]
    gt_images = [cv2.resize(cv2.cvtColor(cv2.imread(f'./data/VOCdevkit/VOC2012/SegmentationClass/{f_name}.png'),cv2.COLOR_BGR2RGB),(config['resolution'],config['resolution']),interpolation=cv2.INTER_NEAREST) for f_name in f_names]
    ground_truth = [_colormap_to_ground_truth(gt_image) for gt_image in gt_images]

    images = np.asarray(images)
    gt_images = np.asarray(gt_images)
    ground_truth = np.asarray(ground_truth)

    np.save(f"./data/{config['resolution']}/images.npy",images)
    np.save(f"./data/{config['resolution']}/gts.npy",ground_truth)
    shutil.copy("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",f"./data/{config['resolution']}/val.txt")
    return


def _colormap_to_ground_truth(image):
    ground_truth = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for rgb, class_index in rgb_to_class_index.items():
        mask = np.all(image == np.array(rgb), axis=-1)
        ground_truth[mask] = class_index
    return ground_truth

def visualize(pred):
    # feature : d, h, w
    pred_label = np.argmax(pred, axis=0)
    img = np.zeros((pred_label.shape[0], pred_label.shape[1], 3), np.uint8)

    for i in range(pred_label.shape[0]):
        for j in range(pred_label.shape[1]):
            img[i,j,:] = VOC_COLORMAP[pred_label[i][j]]
    return img

if __name__ == '__main__':
    dataInit()