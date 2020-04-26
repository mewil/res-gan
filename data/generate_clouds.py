import numpy as np
from os import listdir
from os.path import join
from PIL import Image
from tqdm import tqdm
from re import sub
from argparse import ArgumentParser

from clouds import CloudManager

IMG_SIZE = 406


def generate(filename):
    filenum = int(sub('[^0-9]', '', filename))
    img = Image.open(join(directory, filename))
    pixels = img.getdata()
    if sum(1 for r, g, b in pixels if r == 0 and g == 0 and b == 0) / len(pixels) > 0.20:
        with open('skipped.txt', 'a') as f:
            f.write(filename + '\n')
        return
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img.save(join(directory, '{}.png'.format(filenum)))

    cm = CloudManager()
    co = cm.GetObject(filenum)
    cloud = np.array(co.Colours).reshape([IMG_SIZE, IMG_SIZE, 4])
    cloud = (cloud * 255).astype(np.uint8)
    cloud = Image.fromarray(cloud)
    cloud.save(join(directory, '{}_cloud.png'.format(filenum)))

    img.paste(cloud, (0, 0), cloud.convert('RGBA'))
    img.save(join(directory, '{}_combined.png'.format(filenum)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args()

    directory = args.input_dir
    files = listdir(directory)
    print('Generating', len(files), 'clouds for directory', directory, '...')
    for filename in tqdm(files):
        generate(filename)
    print('Created', len(files), 'clouds in directory', directory)
