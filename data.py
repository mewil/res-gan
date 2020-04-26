import numpy as np
from glob import glob
from cv2 import imread
from os.path import join, basename
from os import listdir

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        all_filenames = list(map(basename, glob(join(self.data_dir, '*.png'))))
        self.imlist = [f for f in all_filenames if '_' not in f]

    def __getitem__(self, index):
        filename = self.imlist[index]
        rgb = imread(join(self.data_dir, filename)).astype(np.float32)
        combined = imread(join(self.data_dir, filename.replace('.png', '_combined.png'))).astype(np.float32)
        return combined.transpose(2, 0, 1), rgb.transpose(2, 0, 1), filename

    def __len__(self):
        return len(self.imlist)
