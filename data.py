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
        self.imlist = list(map(basename, glob(join(self.data_dir, '3band_AOI_1_RIO_img*.png'))))

    def __getitem__(self, index):
        filename = self.imlist[index]
        rgb = imread(join(self.data_dir, filename)).astype(np.float32)
        combined = imread(join(self.data_dir, 'combined_' + filename)).astype(np.float32)
        return combined.transpose(2, 0, 1), rgb.transpose(2, 0, 1)


    def __len__(self):
        return len(self.imlist)

class TestDataset(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.imlist = list(listdir(data_dir))

    def __getitem__(self, index):
        filename = self.imlist[index]
        combined = imread(join(self.data_dir, filename)).astype(np.float32)
        return combined.transpose(2, 0, 1), filename

    def __len__(self):
        return len(self.imlist)
