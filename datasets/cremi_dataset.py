import os
import os.path
import glob
from .listdataset import ListDataset
from .util import split2list
try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)


def make_dataset(dir,split):

    images = []
    for filename in glob.iglob(os.path.join(dir, '*_img1.png')):
        dir_filename = os.path.dirname(filename)
        base_filename = os.path.basename(filename)
        index = base_filename.split('_')[0]
        ref_filename = os.path.join(dir_filename, index+'_img2.png')
        if not os.path.isfile(ref_filename):
            continue
        images.append([filename, ref_filename])
    return split2list(images, split, default_split=0.8)


def cremidataset(root, transform=None, split=None):
    train_list, test_list = make_dataset(root, split)
    train_dataset = ListDataset(train_list, transform)
    test_dataset = ListDataset(test_list, transform)
    return train_dataset, test_dataset







