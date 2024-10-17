import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models
from torch.autograd import Variable


###############################################################################
# Functions
###############################################################################

class PerceptualLoss():
    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def contentFunc(self):
        # Load vgg19 model
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'DiscLoss'

    def initialize(self, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def initialize(self, tensor):
        DiscLoss.initialize(self, tensor)
        self.criterionGAN = GANLoss(use_l1=True, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB):
        return DiscLoss.get_g_loss(self, net, realA, fakeB)

    def get_loss(self, net, realA, fakeB, realB):
        return DiscLoss.get_loss(self, net, realA, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, tensor):
        DiscLossLS.initialize(self, tensor)
        self.LAMBDA = 10

    def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, realA, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty


def init_loss(tensor):
    disc_loss = None
    content_loss = None

    content_loss = PerceptualLoss()
    content_loss.initialize(nn.MSELoss())

    disc_loss = DiscLoss()

    disc_loss.initialize(tensor)
    return disc_loss, content_loss

def get_loss(tensor, netD, realA, fakeB, realB):
    gan_loss, constant_loss = init_loss(tensor)
    loss_D = gan_loss.get_loss(netD, realA, fakeB, realB)
    loss_G = gan_loss.get_g_loss(netD, realA, fakeB) + constant_loss.get_loss(fakeB, realB)
    return loss_D, loss_G

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)
    mu1 = F.conv2d(img1, window, padding=window_size / 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size / 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size / 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size / 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size / 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

import torch
from torch.utils.data import Dataset
from torchvision import transforms as tfs
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import math
        
class ReadConcat(Dataset):
    def __init__(self, opt):
        img_list = []
        for folder in os.listdir(os.path.join(opt.dataroot, 'downscaled_images/downscaled_images')):
            i = 0
            if not os.path.isdir(os.path.join(opt.dataroot, 'downscaled_images/downscaled_images', folder)):
                continue
            for file in os.listdir(os.path.join(opt.dataroot, 'downscaled_images/downscaled_images', folder)):
                if file.endswith('.png'):
                    if i == 10:
                        break
                    img_list.append(os.path.join(folder, file))
                    i += 1
        
        self.img_pathsA = [os.path.join(opt.dataroot,'Set B/Set B', 'type3', k) for k in img_list]
        self.img_pathsB = [os.path.join(opt.dataroot,'downscaled_images/downscaled_images', k) for k in img_list]
        self.img_name = img_list
        self.opt = opt
        transform_list = [tfs.ToTensor(),
                          tfs.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = tfs.Compose(transform_list)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        # img = Image.open(self.img_paths[index]).convert('RGB')
        imgA = Image.open(self.img_pathsA[index]).convert('RGB')
        imgB = Image.open(self.img_pathsB[index]).convert('RGB')
        A = imgA.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        B = imgB.resize((self.opt.loadSizeX, self.opt.loadSizeY), Image.BICUBIC)
        A = self.transform(A)
        B = self.transform(B)

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)


        return {'A': A, 'B': B, 'img_name': img_name}

    def __len__(self):

        return len(self.img_pathsA)

##################################################################################

####################################ImageProcessing##############################


def image_transform(x):
    transform_list = []
    transform_list += [tfs.ToTensor(),
                       tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = tfs.Compose(transform_list)
    return transform(x)    


def image_recovery(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

    
    
def show_compareImage(seta, setb):
    dataset_num = len(seta)
    idxs = np.random.choice(dataset_num, 4, replace=False)
    plt.figure(figsize=(14, 6))
    for i in range(1, 5):
        plt.subplot(2, 4, i)
        plt.imshow(seta[idxs[i-1]])
    for i in range(5, 9):
        plt.subplot(2, 4, i)
        plt.imshow(setb[idxs[i-5]])
    plt.pause(0)


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
        image_pil = Image.fromarray(image_numpy, 'L')
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
#####################################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def update_lr(optim, lr, niter_decay):
    old_lr = lr
    lrd = lr / niter_decay
    lr -= lrd
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    print('update learning rate: %f -> %f' % (old_lr, lr))
    return lr


def save_net(net, checkpoints_dir, net_name, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, net_name)
    check_folder(checkpoints_dir)
    save_path = os.path.join(checkpoints_dir, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        net.cuda()
    print('save_net{}: {}'.format(net_name, save_filename))


def load_net(net, checkpoints_dir, net_name, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, net_name)
    save_path = os.path.join(checkpoints_dir, save_filename)
    net.load_state_dict(torch.load(save_path))
    print('load_net{}: {}'.format(net_name, save_filename))
    
    
def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
import torch
from torch import nn
# import numpy as np

##########################################Generator##############################################

class DeblurGenerator(nn.Module):
    def __init__(self, padding_type='reflect'):
        super(DeblurGenerator, self).__init__()
        # conv-->(downsamping x 2)-->(resnblock x 9)-->(deconv x 2)-->conv-->
        deblur_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True)
        ]

        deblur_model += [
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=True),
            nn.ReLU(True)
        ]

        for i in range(9):
            deblur_model += [
                Resblock(256, padding_type)
            ]

        deblur_model += [
            nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True),
        ]

        deblur_model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*deblur_model)

    def forward(self, x):
        res = x
        out = self.model(x)
        return torch.clamp(out + res, min=-1, max=1)


class Resblock(nn.Module):
    def __init__(self, channel, padding_type):
        super(Resblock, self).__init__()
        # conv-->instanceNorm-->relu-->conv-->instanceNorm-->
        self.conv_block = self.build_conv_block(channel, padding_type)

    def build_conv_block(self, channel, padding_type):
        conv_block = []

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [nn.ZeroPad2d(1)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(channel),
                       nn.ReLU(True)]

        conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        conv_block = self.conv_block(x)
        return conv_block + x


###########################################################################################

####################################Discriminator##########################################


class DeblurDiscriminator(nn.Module):
    def __init__(self):
        super(DeblurDiscriminator, self).__init__()
        # conv-->(downsampling x 2)-->conv-->conv-->
        dis_model = [
            nn.Conv2d(3, 64, 4, 2, padding=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True)
        ]

        dis_model += [
            nn.Conv2d(64, 128, 4, 2, padding=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, padding=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
        ]

        dis_model += [
            nn.Conv2d(256, 512, 4, 1, padding=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, padding=2),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*dis_model)

    def forward(self, x):
        out = self.model(x)

        return out

################################################################################

#####################################test model#################################
# testnet = DeblurGenerator()
# test_x = Variable(torch.zeros(1,3, 72,72))
# print('################test G ####################')
# print('G_input: {}'.format(test_x.shape))
# test_y= testnet(test_x)
# print('G_output: {}'.format(test_y.shape))
#
#
# testnet = DeblurDiscriminator()
# test_x = Variable(torch.zeros(1,3, 72,72))
# print('################test D#####################')
# print('D_input: {}'.format(test_x.shape))
# test_y= testnet.forward(test_x) # 与testnet(test_x)一样
# print('D_output: {}'.format(test_y.shape))

import torch
from torch.autograd import Variable
# import matplotlib.pyplot as plt
# from utils import ReadConcat, image_recovery, update_lr, check_folder, save_image, save_net
# from losses import get_loss
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train(opt, netG, netD, optim_G, optim_D):
    tensor = torch.cuda.FloatTensor
    # lossD_list = []
    # lossG_list = []

    train = ReadConcat(opt)
    trainset = DataLoader(train, batch_size=opt.batchSize, shuffle=True)
    save_img_path = os.path.join('./result', 'train')
    check_folder(save_img_path)

    for e in range(opt.epoch, opt.niter + opt.niter_decay + 1):
        for i, data in enumerate(trainset):
            # set input
            data_A = data['A'] # blur
            data_B = data['B'] #sharp
            # plt.imshow(image_recovery(data_A.squeeze().numpy()))
            # plt.pause(0)
            # print(data_A.shape)
            # print(data_B.shape)

            if torch.cuda.is_available():
                data_A = data_A.cuda(opt.gpu)
                data_B = data_B.cuda(opt.gpu)
            # forward
            realA = Variable(data_A)
            fakeB = netG(realA)
            realB = Variable(data_B)

            # optimize_parameters
            # optimizer netD
            set_requires_grad([netD], True)
            for iter_d in range(1):
                optim_D.zero_grad()
                loss_D, _ = get_loss(tensor, netD, realA, fakeB, realB)
                loss_D.backward()
                optim_D.step()

            # optimizer netG
            set_requires_grad([netD], False)
            optim_G.zero_grad()
            _, loss_G = get_loss(tensor, netD, realA, fakeB, realB)
            loss_G.backward()
            optim_G.step()
            if i % 50 == 0:
                # lossD_list.append(loss_D)
                # lossG_list.append(loss_G)
                print('{}/{}: lossD:{}, lossG:{}'.format(i, e, loss_D, loss_G))

        visul_img = torch.cat((realA, fakeB, realA), 3)
        #print(type(visul_img), visul_img.size())
        visul_img = image_recovery(visul_img)
        #print(visul_img.size)
        save_image(visul_img, os.path.join(save_img_path,'epoch'+str(e)+'.png'))

        if e > opt.niter:
            update_lr(optim_D, opt.lr, opt.niter_decay)
            lr = (optim_G, opt.lr, opt.niter_decay)
            opt.lr = lr

        if e % opt.save_epoch_freq == 0:
            save_net(netG, opt.checkpoints_dir, 'G', e)
            save_net(netD, opt.checkpoints_dir, 'D', e)


import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from utils import ReadConcat, check_folder, image_transform, image_recovery, save_image, PSNR
import os


def test(opt, netG):
    aver_psnr = 0.0
    #aver_ssim = 0.0
    counter = 0

    test = ReadConcat(opt)
    testset = DataLoader(test, batch_size=1, shuffle=False)
    save_path = os.path.join(opt.out_dir, 'test')
    os.makedirs(save_path, exist_ok=True)
    check_folder(save_path)
    netG.eval()
    
    res = 0
    for i,data in enumerate(testset):
        if (res == 99):
            break
        counter = i
        data_A = data['A']  # blur
        data_B = data['B']  # sharp
        if torch.cuda.is_available():
            data_A = data_A.cuda(opt.gpu)
            data_B = data_B.cuda(opt.gpu)
        with torch.no_grad():
            realA = Variable(data_A)
            realB = Variable(data_B)

        fakeB = netG(realA)
        # fakeB = image_recovery(fakeB.squeeze().cpu().detach().numpy())
        # realB = image_recovery(realB.squeeze().cpu().detach().numpy())
        fakeB = image_recovery(fakeB)
        realB = image_recovery(realB)

        aver_psnr += PSNR(fakeB, realB)
        # fakeB = Image.fromarray(fakeB)
        # realB = Image.fromarray(realB)
        # aver_ssim += SSIM(fakeB, realB)
        res += 1 
        # save image
        img_path = data['img_name']
        save_image(fakeB, os.path.join(save_path, img_path[0]))
        print('save successfully {}'.format(save_path))

    aver_psnr /= counter
    # aver_ssim /= counter
    print('PSNR = %f' % (aver_psnr))

import argparse
# from model import DeblurGenerator, DeblurDiscriminator
# from train import train
# from test import test
# from utils import weights_init, load_net
import torch

# parser = argparse.ArgumentParser()

# parser.add_argument('--model', type=str, default='train', help='train or test')
# parser.add_argument('--dataroot', default='./data/dataset/concat_AB', help='path to dataset')
# parser.add_argument('--out_dir', default='./result', help='output direction')
# parser.add_argument('--loadSizeX', type=int, default=360, help='scale images to this size')
# parser.add_argument('--loadSizeY', type=int, default=360, help='scale images to this size')
# parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
# parser.add_argument('--epoch', type=int, default=1, help='the starting epoch count')
# parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
# parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
# parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
# parser.add_argument('--niter', type=int, default=150, help='of iter at starting learning rate')
# parser.add_argument('--niter_decay', type=int, default=150, help='of iter to linearly decay learning rate to zero')
# parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end '
#                                                                    'of epochs')
# parser.add_argument('--checkpoints_dir', default='./checkpoints', help='The direction model saved')
# parser.add_argument('--load_epoch', type=int, default=1, help='load epoch checkpoint')
# parser.add_argument('--gpu', default=0, help='gpu_id')
# parser.add_argument('--no_flip', default=True, help='if specified, do not flip the images for data augmentation')

# opt = parser.parse_args()
class Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args_dict = {
    'model': 'train',
    'dataroot': 'D:\Downloads NA C\MP Project\ML Proej',
    'out_dir': 'D:\Downloads NA C\MP Project\ML Proej',
    'loadSizeX': 360,
    'loadSizeY': 360,
    'fineSize': 256,
    'epoch': 1,
    'lr': 2e-4,
    'batchSize': 1,
    'beta1': 0.5,
    'niter': 1,
    'niter_decay': 0,
    'save_epoch_freq': 1,
    'checkpoints_dir': 'D:\Downloads NA C\MP Project\ML Proej',
    'load_epoch': 1,
    'gpu': 0,
    'no_flip': True
}

opt = Options(**args_dict)


torch.backends.cudnn.benchmark = True

if opt.model == 'train':
    netG = DeblurGenerator().apply(weights_init)
    netD = DeblurDiscriminator().apply(weights_init)
    print( sum(p.numel() for p in netG.parameters()))
    print( sum(p.numel() for p in netD.parameters()))
    if torch.cuda.is_available():
        netG = netG.cuda(opt.gpu)
        netD = netD.cuda(opt.gpu)
    optim_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optim_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    train(opt, netG, netD, optim_G, optim_D)

# import matplotlib.pyplot as plt
# a = ReadConcat(opt, transform=image_transform)
# img = a[10]['A']
# print(type(img))
# print(img.shape)
# #img = image_recovery(img)
# img = img.cpu().float().numpy()
# img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
# img = img.astype(np.uint8)
# plt.imshow(img)
# plt.pause(0)
# # print(img.shape)


# plt.imshow(image_recovery(img))
# plt.pause(0)

