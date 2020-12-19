from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.utils import save_image
from torch.autograd import Variable as V

from models_cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from torch.utils.data import DataLoader
# from Visualizer import Visualizer
import torchvision.transforms as transforms
import Constants
import image_utils
from torch.autograd import Variable
from datasets import *
from scipy.spatial.distance import directed_hausdorff
dataset_name = 'prostate'
# Please specify the ID of graphics cards that you want to use

fold = 'fold2'

def CE_Net_Train():
    NAME = 'CE-Net' + Constants.ROOT.split('/')[-1]

    # run the Visdom
    # viz = Visualizer(env=NAME)

    solver = MyFrame(CE_Net_, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    transforms_ = [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize([0.5],[0.5]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.
    data_loader = DataLoader(
        ImageDataset("../../data/%s/%s" % (dataset_name, fold), transforms_=transforms_),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )


    # start the logging files
    mylog = open('logs/'+ fold + '/' + NAME + '.log', 'w')
    tic = time()
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        for batch in data_loader:

            mask = Variable(batch["B"].type(Tensor))
            img = Variable(batch["A"].type(Tensor))
            img = torch.cat([img, img, img], axis=1)
            # mask = torch.cat([mask, mask, mask], axis=1)
            # print(mask)
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1

        # show the original images, predication and ground truth on the visdom.
        # show_image = (img + 1.6) / 3.2 * 255.
        # viz.img(name='images', img_=show_image[0, :, :, :])
        # viz.img(name='labels', img_=mask[0, :, :, :])
        # viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        print(mylog, '********')
        print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        print(mylog, 'train_loss:', train_epoch_loss)
        print(mylog, 'SHAPE:', Constants.Image_size)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + fold + '/'+ NAME + '_epoch_' + str(epoch) + '.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + fold + '/' + NAME +'_epoch_' + str(epoch) + '.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
        mylog.flush()

    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))



    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * 8
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, img):
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, img):
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, img):
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, img):
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        # print(img)
        # img = cv2.resize(img, (448, 448))
        #
        # img90 = np.array(np.rot90(img))
        # img1 = np.concatenate([img[None], img90[None]])
        # img2 = np.array(img1)[:, ::-1]
        # img3 = np.concatenate([img1, img2])
        # img4 = np.array(img3)[:, :, ::-1]
        # img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        # img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        # img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()  # .squeeze(1)
        # mask1 = mask[:4] + mask[4:, :, ::-1]
        # mask2 = mask1[:2] + mask1[2:, ::-1]
        # mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)

def dice_loss_test(pred, target, smooth=1.):

    pred = torch.cuda.FloatTensor(np.expand_dims(pred, axis=1))
    # print(pred.shape)
    # print(target.shape)
    # target = torch.cuda.FloatTensor(np.expand_dims(target, axis=1))
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()



def dice_loss_test_indv(pred, target, smooth=1.):

    pred = torch.cuda.FloatTensor(np.expand_dims(pred, axis=0))
    # print(pred.shape)
    # print(target.shape)
    pred = torch.unsqueeze(pred, 0)
    target = torch.unsqueeze(target, 0)
    # target = torch.cuda.FloatTensor(np.expand_dims(target, axis=1))
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()
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

def test(model_epoch=None, model_dice=None, model_path=None, im_dim=256, ds_name='test', fold=None, n_classes=6):
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

    print('testing...')
    dice_scores = [];
    ious = []
    hausdorffs = []
    idx = 0
    disc = 20
    solver = TTAFrame(CE_Net_)
    solver.load(model_path)
    for batch in test_dataloader:
        mask = Variable(batch["B"].type(Tensor))
        img = Variable(batch["A"].type(Tensor))
        img_bu = img
        img = torch.cat([img, img, img], axis=1)

        pred = solver.test_one_img_from_path(img)

        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1

        pred_bu = pred

        dh = 0
        count = 0
        for id in range(pred.shape[0]):
            dh += directed_hausdorff(pred[id, :, :], np.squeeze(mask.cpu().numpy(), axis=1)[id, :, :])[0]
            # print(dh)
        if dh/pred.shape[0] != 0.0:
            hausdorffs.append(dh/pred.shape[0])

        idx += 1
        dice_score = float(1 - dice_loss_test(pred, mask).data.cpu().numpy())


        for ic in range(2):
            # only count positive slices
            if 1. in torch.unique(mask[ic, :,:,:]):
                # print(torch.unsqueeze(mask[idx, :,:,:], 0).shape)

                dsc = float(1 - dice_loss_test_indv(pred[ic, :,:], mask[ic, :,:,:]).data.cpu().numpy())
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

        pred = torch.FloatTensor(pred)

        iou = (iou_pytorch(pred, mask).data.cpu().numpy())
        ious.append(iou)
        dice_scores.append(dice_score)

        # print(img_bu.data.shape)
        # print(mask.data.shape)
        # print(torch.unsqueeze(torch.FloatTensor(pred_bu).cuda().data, axis=1).shape)


        img_sample = torch.cat((img_bu.data, mask.data, torch.unsqueeze(torch.FloatTensor(pred_bu).cuda().data, axis=1)), -2)
        save_image(img_sample, "test_images/%s/%s/%s_dice_%s_idx_%s.png" % (ds_name, fold, idx, dice_score, idx),
                   nrow=5,
                   normalize=True)

    dice_score = str(sum(dice_scores) / len(dice_scores))
    std = str(np.sqrt(np.mean(abs(np.array(dice_scores) - np.array(dice_scores).mean()) ** 2)))

    inter_over_union = str(sum(ious) / len(ious))
    std_iou = str(np.sqrt(np.mean(abs(np.array(ious) - np.array(ious).mean()) ** 2)))

    hausdorff = str(sum(hausdorffs) / len(hausdorffs)) 
    std_hausdorff = str(np.sqrt(np.mean(abs(np.array(hausdorffs) - np.array(hausdorffs).mean()) ** 2)))
    print(statistics)
    print('average dice: ' + dice_score)
    print('std dice: ' + std)
    print('average iou: ' + inter_over_union)
    print('std iou: ' + std_iou)
    print('average hausdorff: ' + hausdorff)
    print('std hausdorff: ' + std_hausdorff)

if __name__ == '__main__':
    # print(torch.__version__())
    # CE_Net_Train()
    # test(model_path='./weights_utero/fold1/CE-Net_epoch_41.th', fold='fold1', ds_name='test')
    # test(model_path='./weights/fold2/CE-Net_epoch_40.th', fold='fold2', ds_name='test')
    # test(model_path='./weights_utero/fold3/CE-Net.th', fold='fold3', ds_name='test')
    test(model_path='./weights_utero/fold4/CE-Net.th', fold='fold4', ds_name='test')
