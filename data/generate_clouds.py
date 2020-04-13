import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

from PythonClouds.Clouds import CloudManager

IMG_SIZE = 406


def generate(filename):
    filenum = int(filename.replace('3band_AOI_1_RIO_img', '').replace('.png', ''))
    cm = CloudManager()
    co = cm.GetObject(filenum)
    cloud = co.Colours

    cloud = np.array(co.Colours).reshape([IMG_SIZE, IMG_SIZE, 4])
    cloud = (cloud * 255).astype(np.uint8)

    img = Image.open(join(directory, filename))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img.save(join(directory, filename))

    cloud = Image.fromarray(cloud)        
    cloud.save(join(directory,'cloud_{}'.format(filename)))

    img.paste(cloud, (0, 0), cloud.convert('RGBA'))
    img.save(join(directory,'combined_{}'.format(filename)))



if __name__ == '__main__':
    for directory in ['train', 'val', 'test']:
        files = listdir(directory)
        print('Generating', len(files), 'clouds for directory', directory, '...')
        for filename in tqdm(files):
            generate(filename)
        print('Created', len(files), 'clouds in directory', directory)
