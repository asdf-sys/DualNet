import torch.utils.data as data
import os
import os.path

import numpy as np
try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

def default_loader(imgs_path):
    a = [cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE),(512, 512))[:, :, np.newaxis].astype(np.float32) for img in imgs_path]
    return a

class ListDataset(data.Dataset):
    def __init__(self, path_list, transform=None, loader=default_loader):
        self.path_list = path_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        inputs = self.path_list[index]
        inputs = self.loader(inputs)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        return inputs

    def __len__(self):
        return len(self.path_list)