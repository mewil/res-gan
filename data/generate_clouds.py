import numpy as np
import cv2
from tqdm import tqdm
from os import listdir
from os.path import join

from PythonClouds.Clouds import CloudManager


IMAGE_SIZE = 406

def generate(directory):
    files = listdir(directory)
    print('Creating', len(files), 'clouds for directory', directory, '...')
    for i, filename in tqdm(enumerate(files)):
        cm = CloudManager()
        co = cm.GetObject(i * IMAGE_SIZE)
        cloud = co.Colours

        cloud = np.array(co.Colours).reshape([IMAGE_SIZE, IMAGE_SIZE, 4])
        cloud = (cloud * 255).astype(np.uint8)

        cv2.imwrite(join(directory, 'cloud_{}.png'.format(filename)), cloud)

    print('Created', len(files), 'clouds in directory', directory)


if __name__ == '__main__':
    for directory in ['train', 'val', 'test']:
        generate(directory)
