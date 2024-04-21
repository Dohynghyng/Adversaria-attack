import collections

from utils import data_init, load_config

import os
import cv2
import torch
import numpy as np
from torchvision import transforms

from utils.metrics import ConfusionMatrix, mIoU

class dataLoader():
    def __init__(self):
        data_init.dataInit()
        self.config = load_config.loadConfig()

        self.f_names = []
        with open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r") as f:
            for line in f:
                self.f_names.append(line[:-1])

        # RGB
        self.images = np.load(f"./data/{self.config['resolution']}/images.npy")
        self.gts = np.load(f"./data/{self.config['resolution']}/gts.npy")

        self.transforms = transforms.Compose([
            transforms.Normalize(self.config['mean'], self.config['std']),
        ])

    def adv_batch_load(self, idx):
        test_img = torch.FloatTensor(self.images[idx:idx+1] / 255.0)
        test_img = self.transforms(torch.permute(test_img, (0, 3, 1, 2))).cuda()
        f_name = self.f_names[idx]
        return test_img, f_name

    def val(self,
            model,
            model_name,
            avd_name=None):

        if avd_name!=None:
            model_name = f"{self.config['save_path']}/{avd_name}_{model_name}"

        batch_size = self.config['batch_size']
        cm = ConfusionMatrix(num_classes=21)
        cm.reset()

        img_save_path = './runs/' + model_name + '/'
        if not os.path.isdir(img_save_path):
            os.makedirs(img_save_path)

        for itest in range(0, (self.images.shape[0] // batch_size) + 1):
            print(f"[{model_name}] {itest} / {(self.images.shape[0] // batch_size)}")
            start_num = itest * batch_size
            end_num = (itest + 1) * batch_size
            if end_num > self.images.shape[0]:
                end_num = self.images.shape[0]

            test_img = torch.FloatTensor(self.images[start_num:end_num] / 255.0)
            test_img = self.transforms(torch.permute(test_img,(0,3,1,2))).cuda()
            test_gt, test_name = torch.IntTensor(self.gts[start_num:end_num]).cuda(), self.f_names[start_num:end_num]

            with torch.no_grad():
                pred = model(test_img)
                if isinstance(pred, dict):
                    pred = pred['out']
            cm.update(pred, test_gt)
            # --------img save
            for b in range(end_num - start_num):
                convert_pred = data_init.visualize(pred[b].detach().cpu().numpy())
                cv2.imwrite(img_save_path + f'{test_name[b]}.png', convert_pred)

        model.cpu()
        return mIoU(cm)

    def apply_adv(self, model_name):
        for idx, f_name in enumerate(self.f_names):
            self.images[idx] = cv2.resize(
                cv2.cvtColor(cv2.imread(f"./runs/{self.config['save_path']}/{model_name}/{f_name}.jpg"), cv2.COLOR_BGR2RGB),
                (self.config['resolution'], self.config['resolution']), interpolation=cv2.INTER_LINEAR)


if __name__ == '__main__':
    pass