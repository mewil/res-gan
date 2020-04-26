import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from utils import save_image_from_tensors, get_metrics


def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, batch in enumerate(test_data_loader):
        input_, ground_truth = Variable(batch[0]), Variable(batch[1])
        filename = batch[2][0]
        input_ = F.interpolate(input_, size=256).to(device)
        ground_truth = F.interpolate(ground_truth, size=256).to(device)

        output = gen(input_)

        if epoch % config.snapshot_interval == 0:
            save_image_from_tensors(input_, output, ground_truth, config.out_dir, i, epoch, filename)

        mse, psnr, ssim = get_metrics(output, ground_truth, criterionMSE)
        avg_mse += mse
        avg_psnr += psnr
        avg_ssim += ssim

    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    print('MSE: {:.4f}'.format(avg_mse))
    print('PSNR: {:.4f} dB'.format(avg_psnr))
    print('SSIM: {:.4f} dB'.format(avg_ssim))

    return {
        'epoch': epoch,
        'mse': avg_mse,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
    }
