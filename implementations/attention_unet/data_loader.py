import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import glob


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

class ImageDataset(data.Dataset):
    def __init__(self, root, mode, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
    def __getitem__(self, index):


        img = Image.open(self.files[index % len(self.files)])
        age = int(img.filename.split('_')[-1].split('.')[0])
        age = get_class(age)

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


        return img_A, img_B, age

    def __len__(self):
        return len(self.files)

# class ImageFolder(data.Dataset):
# 	def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
# 		"""Initializes image paths and preprocessing module."""
# 		self.root = root
#
# 		# GT : Ground Truth
# 		self.GT_paths = root[:-1]+'_GT/'
# 		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
# 		self.image_size = image_size
# 		self.mode = mode
# 		self.RotationDegree = [0,90,180,270]
# 		self.augmentation_prob = augmentation_prob
# 		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
#
# 	def __getitem__(self, index):
# 		"""Reads an image from a file and preprocesses it and returns."""
# 		image_path = self.image_paths[index]
# 		filename = image_path.split('_')[-1][:-len(".jpg")]
# 		GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'
#
# 		image = Image.open(image_path)
# 		GT = Image.open(GT_path)
#
# 		aspect_ratio = image.size[1]/image.size[0]
#
# 		Transform = []
#
# 		ResizeRange = random.randint(300,320)
# 		Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
# 		p_transform = random.random()
#
# 		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
# 			RotationDegree = random.randint(0,3)
# 			RotationDegree = self.RotationDegree[RotationDegree]
# 			if (RotationDegree == 90) or (RotationDegree == 270):
# 				aspect_ratio = 1/aspect_ratio
#
# 			Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
#
# 			RotationRange = random.randint(-10,10)
# 			Transform.append(T.RandomRotation((RotationRange,RotationRange)))
# 			CropRange = random.randint(250,270)
# 			Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
# 			Transform = T.Compose(Transform)
#
# 			image = Transform(image)
# 			GT = Transform(GT)
#
# 			ShiftRange_left = random.randint(0,20)
# 			ShiftRange_upper = random.randint(0,20)
# 			ShiftRange_right = image.size[0] - random.randint(0,20)
# 			ShiftRange_lower = image.size[1] - random.randint(0,20)
# 			image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
# 			GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
#
# 			if random.random() < 0.5:
# 				image = F.hflip(image)
# 				GT = F.hflip(GT)
#
# 			if random.random() < 0.5:
# 				image = F.vflip(image)
# 				GT = F.vflip(GT)
#
# 			Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)
#
# 			image = Transform(image)
#
# 			Transform =[]
#
#
# 		Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
# 		Transform.append(T.ToTensor())
# 		Transform = T.Compose(Transform)
#
# 		image = Transform(image)
# 		GT = Transform(GT)
#
# 		Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# 		image = Norm_(image)
#
# 		return image, GT
#
# 	def __len__(self):
# 		"""Returns the total number of font files."""
# 		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, mode, num_workers=2, transforms=None, shuffle=True):
	"""Builds and returns Dataloader."""
	
	dataset = ImageDataset(root = image_path, mode=mode, transforms_=transforms)

	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=shuffle,
								  num_workers=num_workers)
	return data_loader
