import os
import random
import shutil
import yaml
from attrdict import AttrMap
from tqdm import tqdm
from datetime import datetime
import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from data import Dataset
from models.generator import UNet
from models.discriminator import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from validate import test
from log_report import LogReport
from log_report import TestReport


def train(config):
    gpu_manage(config)

    train_dataset = Dataset(config.train_dir)
    val_dataset = Dataset(config.val_dir)
    training_data_loader = DataLoader(dataset=train_dataset,
                                      num_workers=config.threads,
                                      batch_size=config.batchsize,
                                      shuffle=True)
    val_data_loader = DataLoader(dataset=val_dataset,
                                 num_workers=config.threads,
                                 batch_size=config.test_batchsize,
                                 shuffle=False)

    gen = UNet(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    if config.dis_init is not None:
        param = torch.load(config.dis_init)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    real_a = torch.FloatTensor(config.batchsize, config.in_ch, 256, 256)
    real_b = torch.FloatTensor(config.batchsize, config.out_ch, 256, 256)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.cuda:
        gen = gen.cuda(0)
        dis = dis.cuda(0)
        criterionL1 = criterionL1.cuda(0)
        criterionMSE = criterionMSE.cuda(0)
        criterionSoftplus = criterionSoftplus.cuda(0)
        real_a = real_a.cuda(0)
        real_b = real_b.cuda(0)

    real_a = Variable(real_a)
    real_b = Variable(real_b)

    logreport = LogReport(log_dir=config.out_dir)
    testreport = TestReport(log_dir=config.out_dir)

    # main
    for epoch in range(1, config.epoch + 1):
        print('Epoch', epoch, datetime.now())
        for iteration, batch in enumerate(tqdm(training_data_loader)):
            real_a, real_b = batch[0], batch[1]
            real_a = F.interpolate(real_a, size=256).to(device)
            real_b = F.interpolate(real_b, size=256).to(device)
            fake_b = gen.forward(real_a)

            # Update D
            opt_dis.zero_grad()

            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            batchsize, _, w, h = pred_fake.size()

            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h
            loss_d = loss_d_fake + loss_d_real
            loss_d.backward()

            if epoch % config.minimax == 0:
                opt_dis.step()

            # Update G
            opt_gen.zero_grad()
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)

            loss_g_gan = torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h
            loss_g = loss_g_gan + criterionL1(fake_b, real_b) * config.lamb
            loss_g.backward()

            opt_gen.step()

            if iteration % 100 == 0:
                logreport({
                    'epoch': epoch,
                    'iteration': len(training_data_loader) * (epoch - 1) + iteration,
                    'gen/loss': loss_g.item(),
                    'dis/loss': loss_d.item(),
                })

        with torch.no_grad():
            log_test = test(config, val_data_loader, gen, criterionMSE, epoch)
            testreport(log_test)

        if epoch % config.snapshot_interval == 0:
            checkpoint(config, epoch, gen, dis)

        logreport.save_lossgraph()
        testreport.save_lossgraph()

    print('Done', datetime.now())


if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train(config)
