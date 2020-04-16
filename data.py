# import random
import numpy as np
# import pickle
# import os
from glob import glob
from cv2 import imread
from os.path import join, basename

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.imlist = list(map(basename, glob(join(self.data_dir, '3band_AOI_1_RIO_img*.png'))))

    def __getitem__(self, index):
        filename = self.imlist[index]
#         print(filename)
        rgb = imread(join(self.data_dir, filename)).astype(np.float32)
        combined = imread(join(self.data_dir, 'combined_' + filename)).astype(np.float32)
        
#         print(combined.shape, rgb.shape)
        return combined, rgb
        # nir = cv2.imread(join(self.config.rgbnir_dir, 'NIR', str(self.imlist[index])), 0).astype(np.float32)
        # cloud = cv2.imread(self.cloud_files[random.randrange(self.n_cloud)], -1).astype(np.float32)

        # alpha = cloud[:, :, 3] / 255.
        # alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
        # cloud_rgb = (1. - alpha) * rgb + alpha * cloud[:, :, :3]
        # cloud_rgb = np.clip(cloud_rgb, 0., 255.)

        # cloud_mask = cloud[:, :, 3]

        # x = np.concatenate((cloud_rgb, nir[:, :, None]), axis=2)
        # t = np.concatenate((rgb, cloud_mask[:, :, None]), axis=2)

        # x = x / 127.5 - 1
        # t = t / 127.5 - 1

        # x = x.transpose(2, 0, 1)
        # t = t.transpose(2, 0, 1)

        # return x, t

    def __len__(self):
        return len(self.imlist)

# class TestDataset(data.Dataset):
#     def __init__(self, test_dir):
#         super().__init__()
#         self.test_dir = test_dir
        
#         self.test_files = glob.glob(os.path.join(test_dir, 'RGB', '*.png'))

#     def __getitem__(self, index):
#         filename = os.path.basename(self.test_files[index])
#         cloud_rgb = cv2.imread(os.path.join(self.test_dir, 'RGB', filename), 1).astype(np.float32)
#         nir = cv2.imread(os.path.join(self.test_dir, 'NIR', filename), 0).astype(np.float32)

#         x = np.concatenate((cloud_rgb, nir[:, :, None]), axis=2)

#         x = x / 127.5 - 1

#         x = x.transpose(2, 0, 1)

#         return x, filename

#     def __len__(self):

#         return len(self.test_files)
