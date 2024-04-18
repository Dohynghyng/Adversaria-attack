from utils import data_init, load_config

import numpy as np


class dataLoader():
    def __init__(self):
        data_init.dataInit()
        self.config = load_config.loadConfig()

        self.f_names = []
        with open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r") as f:
            for line in f:
                self.f_names.append(line[:-1])
        self.images = np.load(f"./data/{self.config['resolution']}/images.npy")
        self.gts = np.load(f"./data/{self.config['resolution']}/gt.npy")


if __name__ == '__main__':
    pass