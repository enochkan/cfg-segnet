import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def get_class(age):
    if 1 <= age <= 3:
        return 0
    elif 4 <= age <= 6:
        return 1
    elif 7 <= age <= 9:
        return 2
    elif 10 <= age <= 12:
        return 3
    elif 13 <= age <= 15:
        return 4
    else:
        return 5


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        age = int(img.filename.split('_')[-1].split('.')[0])

        w, h = img.size

        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, :])
            img_B = Image.fromarray(np.array(img_B)[:, :])

        # img_A_bw = np.array(img_A)
        # stacked = np.maximum(img_A_bw[:, :], np.array(img_B)[:, :])
        # img_B = Image.fromarray(stacked)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "label": get_class(age)}

    def __len__(self):
        return len(self.files)
