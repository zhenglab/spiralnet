import os 
import glob
import scipy
import torch
import random
import numpy as np
import cv2
import torchvision.transforms.functional as F

from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from .utils import random_crop, center_crop, side_crop 
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, input_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.input_size = config.INPUT_SIZE
        self.center = config.CENTER
        self.model = config.MODEL
        self.augment = augment
        self.training = training
        self.data = self.load_flist(input_flist)
        self.side = config.SIDE
        self.mean = config.MEAN
        self.std = config.STD
        self.count = 0
        self.pos = None
        self.batchsize = config.BATCH_SIZE
        self.catmask = config.CATMASK
        self.datatype = config.DATATYPE
        self.known_mask = config.MODE == 2
        if self.datatype == 2:
            self.scence_width = 512
            self.scence_height = 256
        if self.known_mask:
            self.mask_file = self.load_flist(mask_flist)
            print("mask_file: ",len(self.mask_file))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.load_item(index)

        return item

    def resize(self, img, width, height):
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        return img

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name), self.data[index]
        

    def load_item(self, index):
        data = imread(self.data[index])
        self.seq = index

        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)
        if self.datatype == 1:
            data = self.resize(data, self.input_size, self.input_size)
            half_data = self.resize(data, self.input_size//2, self.input_size//2)
        if self.datatype == 2:
            data = self.resize(data, self.scence_width, self.scence_height)
            half_data = self.resize(data, self.scence_width//4, self.scence_height//2)

        pdata, pos, mask = self.cpimage(data)
        
        half_fmask = self.resize(mask, half_data.shape[1], half_data.shape[0])       
        fmask_data = mask
        z = torch.FloatTensor(np.random.normal(0, 1, (3, half_data.shape[1],half_data.shape[0])))
        
        self.count += 1
        if self.count == self.batchsize and not self.known_mask:
            self.count = 0
        
        if self.augment and np.random.binomial(1, 0.5) > 0:
            half_data = half_data[:, ::-1, ...]
            data = data[:, ::-1, ...]
            pdata = pdata[:, ::-1, ...]
            fmask_data = fmask_data[:, ::-1, ...]
            half_fmask = half_fmask[:, ::-1, ...]
            
        if self.model == 1 :
            half = self.to_tensor(half_data)
        else:
            gdata = self.to_tensor(data)

        if self.datatype == 1:
            halfpdata = pdata
        
        if self.datatype == 2:
            halfpdata = self.resize(pdata, self.input_size//2, self.input_size//2)
        
        return (half if self.model == 1 else gdata), self.to_tensor(halfpdata), self.to_tensor(pdata), torch.IntTensor(pos),\
                self.to_tensor(fmask_data), self.to_tensor(half_fmask),\
                (half if self.model == 1 else gdata) * \
                (1 - self.to_tensor(half_fmask if self.model == 1 else fmask_data)), z
  
        
    def img_resize(self, img, width, height, centerCrop=False):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)

        return img

    def locate_mask(self, data, mask):
        height, width = data.shape[0:2]
        coord = 0
        for i in range(width):
            for j in range(height):
                if (mask[i][j] != 0):
                    coord = (j,i)
                    break
            if (mask[i][j] != 0):
                break

        inner_img = data[i:i+128, j:j+128]

        return inner_img, coord
    
    
    def dealimage(self, data, mask):
        rc, pos = self.locate_mask(data, mask)
        return rc, pos

    def cpimage(self, data):
        if self.known_mask:
            mask = imread(self.mask_file[self.seq])
            rc, pos = self.dealimage(data, mask)
            self.pos = pos
        rc, pos, mask = random_crop(data, int(data.shape[1]/2), self.datatype, self.count, self.pos,self.known_mask)
        self.pos = pos
        return rc, pos, mask
    
    def gray_fmap(self, fmap_data):
        fmap_data = cv2.cvtColor(fmap_data, cv2.COLOR_BGR2GRAY)
        fmap_data[fmap_data < fmap_data.mean()+15] = 0
        fmap_data = cv2.equalizeHist(fmap_data)
        
        return fmap_data


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            # print(flist)
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                # try:
                # print(np.genfromtxt(flist, dtype=np.str, encoding='utf-8'))
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                # except:
                    # print(11, flist)
                #    return [flist]
        
        return []
    
    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
    
    def to_tensor_norm(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        img_t = F.normalize(img_t, self.mean, self.std)  # 输入mean 和 std
        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
