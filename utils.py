import os
import cv2
import random
import numpy as np
import torch
from torch.backends import cudnn
from skimage.measure import compare_ssim as SSIM


def gpu_manage(config):
    if config.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_ids))
        config.gpu_ids = list(range(len(config.gpu_ids)))
        print(os.environ['CUDA_VISIBLE_DEVICES'])

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')


def save_image(out_dir, x, num, epoch, filename=None):
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename)
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    cv2.imwrite(test_path, x)


def checkpoint(config, epoch, gen, dis):
    model_dir = os.path.join(config.out_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    net_gen_model_out_path = os.path.join(model_dir, 'gen_model_epoch_{}.pth'.format(epoch))
    net_dis_model_out_path = os.path.join(model_dir, 'dis_model_epoch_{}.pth'.format(epoch))
    torch.save(gen.state_dict(), net_gen_model_out_path)
    torch.save(dis.state_dict(), net_dis_model_out_path)
    print('Checkpoint saved to {}'.format(model_dir))


def make_manager():
    if not os.path.exists('.job'):
        os.makedirs('.job')
        with open('.job/job.txt', 'w') as f:
            f.write('0')


def job_increment():
    with open('.job/job.txt', 'r') as f:
        n_job = f.read()
        n_job = int(n_job)
    with open('.job/job.txt', 'w') as f:
        f.write(str(n_job + 1))

    return n_job


def save_image_from_tensors(input_, output, ground_truth, out_dir, i, epoch, filename):
    h = 1
    w = 3
    c = 3
    p = 256

    img = np.zeros((h, w, c, p, p))
    combined = input_.cpu().numpy()[0]
    result = output.cpu().numpy()[0]
    ground = ground_truth.cpu().numpy()[0]
    img[0, 0, :, :, :] = combined
    img[0, 1, :, :, :] = result
    img[0, 2, :, :, :] = ground
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape((h * p, w * p, c))

    save_image(out_dir, img, i, epoch, filename=filename)


def get_metrics(output, ground_truth, criterionMSE):
    img1 = np.tensordot(output.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    img2 = np.tensordot(ground_truth.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)

    mse = criterionMSE(output, ground_truth).item()
    psnr = 10 * np.log10(1 / mse)
    ssim = SSIM(img1, img2)
    return mse, psnr, ssim
