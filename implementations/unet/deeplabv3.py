import argparse
import os
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from implementations import dlv3_network as network
# from segmentation_unet import UNet

import torch

# model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=False)


def dice_loss_test(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="pancreas_conditioned", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=7, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                    choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                             'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                             'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
parser.add_argument("--separable_conv", action='store_true', default=False,
                    help="apply separable conv to decoder and aspp")
parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])
parser.add_argument("--seg_lr", type=float, default=1e-2, help="segmentor adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=6, help="number of image channels")
parser.add_argument("--num_classes", type=int, default=1,
                        help="num classes (default: None)")
parser.add_argument("--n_critic", type=int, default=1, help="n critic number- how many iter before training segmentor")
parser.add_argument("--sample_interval", type=int, default=151, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s/dlv3_%s" % (opt.dataset_name, opt.model), exist_ok=True)
os.makedirs("saved_models/dlv3_%s%s" % (opt.dataset_name, opt.model), exist_ok=True)

model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

segmentor = model_map[opt.model](num_classes=opt.num_classes, output_stride=opt.output_stride)

# Loss functions
criterion = torch.nn.BCEWithLogitsLoss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

if cuda:
    segmentor = segmentor.cuda()
    criterion.cuda()

if opt.epoch != 0:
    # Load pretrained models
    segmentor.load_state_dict(torch.load("saved_models/%s/segmentor_%d.pth" % (opt.dataset_name, opt.epoch)))

# Optimizers
optimizer_S = torch.optim.Adam(segmentor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize([0.5],[0.5]),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=5,
    shuffle=False,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    print('validating...')
    # calculate dice score (average)
    dice_scores = []; idx=0
    for batch in tqdm(val_dataloader):

        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))

        mask = segmentor(real_B)

        mask[mask <= 0.5] = 0
        mask[mask > 0.5] = 1

        idx += 1
        dice_score = float(1-dice_loss_test(mask, real_A).data.cpu().numpy())

        dice_scores.append(dice_score)
        img_sample = torch.cat((real_B.data, real_A.data, mask.data), -2)
        save_image(img_sample, "images/%s/dlv3_%s/%s_dice_%s_idx_%s.png" % (opt.dataset_name, opt.model, batches_done, dice_score, idx), nrow=5,
                   normalize=True)

    dice_score = str(sum(dice_scores)/len(dice_scores))


    print('average validation dice score: ' + dice_score)

# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for idx, batch in enumerate(dataloader):
        # Model inputs
        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))
        optimizer_S.zero_grad()
        mask = segmentor(real_B)

        loss = criterion(mask, real_A)

        # Total loss
        loss.backward()

        optimizer_S.step()


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
            "\r[Epoch %d/%d] [Batch %d/%d] [bce loss: %s] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                idx,
                len(dataloader),
                loss.item(),
                time_left,
            ), end=""
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        torch.save(segmentor.state_dict(), "saved_models/%s/segmentor_%d.pth" % (opt.dataset_name, epoch))

22222
22222
22222
22222
22222
