import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

from segmentation_unet import UNet

from utils import dice_loss

import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.spatial.distance import directed_hausdorff


def dice_loss_test(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--fold", type=str, default='fold4', help="cv fold")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="test", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--seg_lr", type=float, default=1e-2, help="segmentor adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=2, help="number of image channels")
    parser.add_argument("--n_classes", type=int, default=6, help="number of image channels")
    parser.add_argument("--save_best_model_only", type=int, default=0, help="saving best model only")
    parser.add_argument("--n_critic", type=int, default=1, help="n critic number- how many iter before training segmentor")
    parser.add_argument("--sample_interval", type=int, default=1576, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    # 1576, 1577, 1585, 1576

    os.makedirs("images/%s/%s" % (opt.dataset_name, opt.fold), exist_ok=True)
    os.makedirs("saved_models/%s/%s" % (opt.dataset_name, opt.fold), exist_ok=True)

    best_model = None
    best_dice = 0.0

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    criterion_dice = dice_loss()
    criterion_bce = torch.nn.BCEWithLogitsLoss()

    cuda = True if torch.cuda.is_available() else False

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_image_pixel = 200
    # lambda_aux = 30
    lambda_segmentation = 50

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    segmentor = UNet(n_channels=1, n_classes=1)
    discriminator = Discriminator(opt.n_classes)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        segmentor = segmentor.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
        auxiliary_loss.cuda()
        criterion_dice.cuda()
        criterion_bce.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/%s/generator_%d.pth" % (opt.dataset_name, opt.fold, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/%s/discriminator_%d.pth" % (opt.dataset_name, opt.fold, opt.epoch)))
        segmentor.load_state_dict(torch.load("saved_models/%s/%s/segmentor_%d.pth" % (opt.dataset_name, opt.fold,  opt.epoch)))

    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        # segmentor.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters(), segmentor.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize([0.5],[0.5]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset("../../data/%s/%s" % (opt.dataset_name, opt.fold), transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        ImageDataset("../../data/%s/%s" % (opt.dataset_name, opt.fold), transforms_=transforms_, mode="val"),
        batch_size=5,
        shuffle=False,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def sample_images(batches_done):
        print('validating...')

        # calculate dice score (average)
        dice_scores = []; idx=0
        for batch in tqdm(val_dataloader):

            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))

            labels_base = Variable(batch["label"])
            labels_embed = labels_base.float().clone().detach()

            mask_real = segmentor(real_B)

            mask_real_stacked = np.empty(
                (tuple(mask_real.shape)[0], 2, tuple(mask_real.shape)[2], tuple(mask_real.shape)[3]))
            for i, mask in enumerate(mask_real.cpu()):
                msk = np.array(mask.detach())
                label = np.tile(labels_embed.cpu()[i], reps=(tuple(mask_real.shape)[1:]))
                stacked = np.squeeze(np.stack((msk, label), axis=1))
                mask_real_stacked[i] = stacked
            mask_real_stacked_tensor = Variable(torch.tensor(mask_real_stacked).float()).cuda()

            fake_B = generator(mask_real_stacked_tensor)

            mask = segmentor(real_B)

            mask[mask <= 0.5] = 0
            mask[mask > 0.5] = 1

            idx += 1
            dice_score = float(1-dice_loss_test(mask, real_A).data.cpu().numpy())

            dice_scores.append(dice_score)
            img_sample = torch.cat((real_B.data, fake_B.data, real_A.data, mask.data), -2)
            save_image(img_sample, "images/%s/%s/%s_dice_%s_idx_%s.png" % (opt.dataset_name, opt.fold, batches_done, dice_score, idx), nrow=5,
                    normalize=True)

        dice_score = str(sum(dice_scores)/len(dice_scores))


        print('average validation dice score: ' + dice_score)
        return dice_score

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        for idx, batch in enumerate(dataloader):
            # Model inputs
            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))

            # *** NEW ***
            labels_base = np.array(batch["label"])
            labels = Variable(torch.LongTensor(labels_base)).cuda()
            labels_embed = Variable(torch.tensor(labels_base).float()).cuda()

            gen_labels_base = np.random.randint(0, opt.n_classes, opt.batch_size)
            gen_labels = Variable(torch.LongTensor(gen_labels_base)).cuda()
            gen_labels_embed = Variable(torch.LongTensor(gen_labels_base).float()).cuda()

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()

            # Segmentation, train with real then train with fake
            mask_real = segmentor(real_B)

            mask_real_stacked = np.empty((tuple(mask_real.shape)[0], 2, tuple(mask_real.shape)[2], tuple(mask_real.shape)[3]))

            for i, mask in enumerate(mask_real.cpu()):
                msk = np.array(mask.detach())
                label = np.tile(labels_embed.cpu()[i], reps=(tuple(mask_real.shape)[1:]))
                stacked = np.squeeze(np.stack((msk, label), axis=1))
                mask_real_stacked[i] = stacked

            mask_real_stacked_tensor = Variable(torch.tensor(mask_real_stacked).float()).cuda()

            # GAN loss
            fake_B = generator(mask_real_stacked_tensor)

            mask_fake = segmentor(fake_B)

            #bce loss
            loss_bce_real = criterion_bce(mask_real, real_A)

            loss_bce_fake = criterion_bce(mask_fake, real_A)

            loss_bce = (loss_bce_fake + loss_bce_real)/2

            pred_fake, pred_labels = discriminator(fake_B.detach(), real_A)

            # Adversarial loss
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Auxiliary loss
            loss_aux = auxiliary_loss(pred_labels, gen_labels)

            # Total loss
            loss_G = (loss_GAN + loss_aux)/2 + lambda_image_pixel * loss_pixel + lambda_segmentation * loss_bce

            loss_G.backward()

            optimizer_G.step()


            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real, real_aux = discriminator(real_B, real_A)

            loss_real = (criterion_GAN(pred_real, valid) + auxiliary_loss(real_aux, labels))/2

            # Fake loss
            pred_fake, fake_aux = discriminator(fake_B.detach(), real_A)

            loss_fake = (criterion_GAN(pred_fake, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            loss_D.backward()
            optimizer_D.step()


            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + idx
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print(
                "\r[Epoch %d/%d] [CV %s] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f, aux: %f, acc: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    opt.fold,
                    idx,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    loss_aux.item(),
                    d_acc.item(),
                    time_left,
                ), end=""
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                ds = sample_images(batches_done)
                if float(ds) > best_dice and opt.save_best_model_only:
                    # save model if dice score improves
                    best_dice = float(ds)
                    torch.save(generator.state_dict(), "saved_models/%s/%s/generator_%d_dice_%s.pth" % (opt.dataset_name, opt.fold, epoch, str(ds)))
                    torch.save(discriminator.state_dict(), "saved_models/%s/%s/discriminator_%d_dice_%s.pth" % (opt.dataset_name, opt.fold, epoch, str(ds)))
                    torch.save(segmentor.state_dict(), "saved_models/%s/%s/segmentor_%d_dice_%s.pth" % (opt.dataset_name, opt.fold, epoch, str(ds)))
                    print('Saved best model')
                else:
                    torch.save(generator.state_dict(), "saved_models/%s/%s/generator_%d_dice_%s.pth" % (
                    opt.dataset_name, opt.fold, epoch, str(ds)))
                    torch.save(discriminator.state_dict(), "saved_models/%s/%s/discriminator_%d_dice_%s.pth" % (
                    opt.dataset_name, opt.fold, epoch, str(ds)))
                    torch.save(segmentor.state_dict(), "saved_models/%s/%s/segmentor_%d_dice_%s.pth" % (
                    opt.dataset_name, opt.fold, epoch, str(ds)))
                    print('Saved model')

        # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        #     # Save model checkpoints
        #     torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        #     torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
        #     torch.save(segmentor.state_dict(), "saved_models/%s/segmentor_%d.pth" % (opt.dataset_name, epoch))


SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.type(torch.IntTensor)
    labels = labels.type(torch.IntTensor)
    # print(outputs)
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch




def test(model_epoch=None, model_dice=None, im_dim=256, ds_name='test', fold='fold2', n_classes=6):
    statistics = {}

    os.makedirs("test_images/%s/%s" % (ds_name, fold), exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    transforms_ = [
        transforms.Resize((im_dim, im_dim), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize([0.5],[0.5]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    test_dataloader = DataLoader(
        ImageDataset("../../data/%s/%s" % (ds_name, fold), transforms_=transforms_, mode="test"),
        batch_size=2,
        shuffle=False,
        num_workers=1,
    )
    d_name = 'prostate'
    generator = GeneratorUNet().cuda()
    generator.load_state_dict(torch.load('saved_models_prostate/'+d_name+'/' + fold + '/' +'generator_'+str(model_epoch)+'_dice_'+str(model_dice)+'.pth'))
    generator.eval()
    segmentor = UNet(n_channels=1, n_classes=1).cuda()
    segmentor.load_state_dict(torch.load('saved_models_prostate/'+d_name+'/'+ fold + '/' + 'segmentor_'+str(model_epoch)+'_dice_'+str(model_dice)+'.pth'))
    segmentor.eval()
    discriminator = Discriminator(n_classes).cuda()
    discriminator.load_state_dict(torch.load('saved_models_prostate/'+d_name+'/'+fold + '/' + 'discriminator_'+str(model_epoch)+'_dice_'+str(model_dice)+'.pth'))
    discriminator.eval()
    
    print('testing...')
    dice_scores = []; idx=0
    ious = []
    hausdorffs = []
    for batch in tqdm(test_dataloader):

        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))

        labels_base = Variable(batch["label"])
        labels_embed = labels_base.float().clone().detach()

        mask_real = segmentor(real_B)

        mask_real_stacked = np.empty(
            (tuple(mask_real.shape)[0], 2, tuple(mask_real.shape)[2], tuple(mask_real.shape)[3]))
        for i, mask in enumerate(mask_real.cpu()):
            msk = np.array(mask.detach())
            label = np.tile(labels_embed.cpu()[i], reps=(tuple(mask_real.shape)[1:]))
            stacked = np.squeeze(np.stack((msk, label), axis=1))
            mask_real_stacked[i] = stacked
        mask_real_stacked_tensor = Variable(torch.tensor(mask_real_stacked).float()).cuda()

        fake_B = generator(mask_real_stacked_tensor)

        mask = segmentor(real_B)

        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1

        dh = 0

        for id in range(mask.shape[0]):
            dh += directed_hausdorff(np.squeeze(mask.detach().cpu().numpy(), axis=1)[id, :, :], np.squeeze(real_A.cpu().numpy(), axis=1)[id, :, :])[0]
        if dh/mask.shape[0] != 0.0:
            hausdorffs.append(dh / mask.shape[0])

        idx += 1
        dice_score = float(1-dice_loss_test(mask, real_A).data.cpu().numpy())

        for ic in range(2):
            # only count positive slices
            if 1. in torch.unique(real_A[ic, :,:,:]):
                # print(torch.unsqueeze(mask[idx, :,:,:], 0).shape)
                dsc = float(1 - dice_loss_test(torch.unsqueeze(mask[ic, :,:,:], 0), torch.unsqueeze(real_A[ic, :,:,:], 0)).data.cpu().numpy())
                age = int(batch["label"][ic].cpu().detach().int())
                age = str(age)
                # if age not in statistics.keys():
                #     statistics[age] = dice_score
                # else:
                #     statistics[age] += dice_score
                #
                # if age + '_count' not in statistics.keys():
                #     statistics[age + '_count'] = 1
                # else:
                #     statistics[age + '_count'] += 1

                if age + '_elements' not in statistics.keys():
                    statistics[age + '_elements'] = []
                else:
                    statistics[age + '_elements'].append(dsc)

        iou = (iou_pytorch(mask, real_A).data.cpu().numpy())
        # print(iou)

        ious.append(iou)
        dice_scores.append(dice_score)
        img_sample = torch.cat((real_B.data, fake_B.data, real_A.data, mask.data), -2)
        save_image(img_sample, "test_images/%s/%s/%s_dice_%s_idx_%s.png" % (ds_name,fold, idx, dice_score, idx), nrow=5,
                normalize=True)


    dice_score = str(sum(dice_scores) / len(dice_scores))
    std = str(np.sqrt(np.mean(abs(np.array(dice_scores) - np.array(dice_scores).mean()) ** 2)))

    inter_over_union = str(sum(ious) / len(ious))
    std_iou = str(np.sqrt(np.mean(abs(np.array(ious) - np.array(ious).mean()) ** 2)))

    hausdorff = str(sum(hausdorffs)/len(hausdorffs))
    std_hausdorff = str(np.sqrt(np.mean(abs(np.array(hausdorffs) - np.array(hausdorffs).mean()) ** 2)))

    print(statistics)
    print('average dice: ' + dice_score)
    print('std dice: ' + std)
    print('average iou: ' + inter_over_union)
    print('std iou: '+std_iou)
    print('average hausdorff: ' + hausdorff)
    print('std hausdorff: ' + std_hausdorff)
    # print('average ious: '+inter_over_union)
    # return dice_score

if __name__ == '__main__':
  
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # run()
    # fold 1
    # test(model_epoch=41, model_dice=0.6045084593817591, ds_name='test', fold='fold1')
    # fold 2
    # test(model_epoch=41, model_dice=0.7994792246725411, ds_name='test', fold='fold2')

    # fold 3
    # test(model_epoch=43, model_dice=0.8022018915042282, ds_name='test', fold='fold3')
    # fold 4
    # test(model_epoch=45, model_dice=0.6544647041708231, ds_name='test', fold='fold4')


    # prostate
    # test(model_epoch=41, model_dice=0.9746659077393512, ds_name='prostate', fold='fold1')
    # test(model_epoch=45, model_dice=0.8309877689927816, ds_name='prostate', fold='fold2')
    # test(model_epoch=44, model_dice=0.9688990495799642, ds_name='prostate', fold='fold3')
    test(model_epoch=32, model_dice=0.9404379718626539, ds_name='prostate', fold='fold4')
