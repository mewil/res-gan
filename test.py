import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import Dataset
from utils import gpu_manage, save_image
from models.generator import UNet
from utils import save_image_from_tensors, get_metrics


def predict(config, args):
    gpu_manage(args)
    dataset = Dataset(args.test_dir)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    gen = UNet(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=args.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)
    criterionMSE = nn.MSELoss()

    if args.cuda:
        gen = gen.cuda(0)
        criterionMSE = criterionMSE.cuda(0)

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            input_, ground_truth = Variable(batch[0]), Variable(batch[1])
            filename = batch[2][0]

            output = gen(input_)

            save_image_from_tensors(input_, output, ground_truth, config.out_dir, i, epoch, filename)
            mse, psnr, ssim = get_metrics(output, ground_truth, criterionMSE)
            print(filename)
            print('MSE: {:.4f}'.format(mse))
            print('PSNR: {:.4f} dB'.format(psnr))
            print('SSIM: {:.4f} dB'.format(ssim))

            avg_mse += mse
            avg_psnr += psnr
            avg_ssim += ssim

    avg_mse = avg_mse / len(data_loader)
    avg_psnr = avg_psnr / len(data_loader)
    avg_ssim = avg_ssim / len(data_loader)

    print('Average MSE: {:.4f}'.format(avg_mse))
    print('Average PSNR: {:.4f} dB'.format(avg_psnr))
    print('Average SSIM: {:.4f} dB'.format(avg_ssim))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = AttrMap(config)

    predict(config, args)
