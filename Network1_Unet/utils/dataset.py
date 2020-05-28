import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

#for roots,dirs,files in os.walk('/home/nihar/Desktop/Pytorch-UNet-master/Training/Damaged'):
#                print(roots,len(dirs),len(files))

#Dataset = '/home/nihar/Desktop/Pytorch-UNet-master/Training/Damaged'

class BasicDataset(Dataset):
    def __init__(self, imgs_dir,mask_dir, scale=0.5):
        self.imgs_dir = imgs_dir
        self.ids = [];
        self.scale = scale
        for root,dirs,files in os.walk(imgs_dir):
          if(len(files)>1 and 'AP.jpg' in files and 'Ap_Vertebra.png' in files):
            a = Image.open(root+'/'+files[files.index('AP.jpg')])
            n = np.shape(a)
            if (np.size(n) == 3):
              temp = [(root + '/' +files[files.index('AP.jpg')]),(root + '/' +files[files.index('Ap_Vertebra.png')])]
              self.ids.append(temp)
        # self.ids = [((root+'/'+files[0]),(root +'/'+ files[1])) for root,dirs,files in os.walk(imgs_dir)
        #             if len(files) > 1
        #             if (files[0].startswith('AP.jpg'))]
        #assert 0 < scale <= 1, 'Scale must be between 0 and 1'
	
                

        #self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #            if not file.startswith('.')]
        
    def __len__(self):
        return len(self.ids)

    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idxi, idxm = self.ids[i]
        mask_file = glob(idxm)
        img_file = glob(idxi)
       # print("hi")
	

        
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        # print(np.shape(mask))
        # print(np.shape(img))
        mask = mask.convert('L')
        # print(img.size)
        # print(mask.size)
        img = img.resize((1000,700))
        mask = mask.resize(img.size)
        assert img.size == mask.size, \
            f'Image and mask {self.ids[i]} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # print("hi")
        # print(np.shape(mask))
        # print(np.shape(img))

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

   
