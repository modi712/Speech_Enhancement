import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

#for roots,dirs,files in os.walk('/home/nihar/Desktop/Pytorch-UNet-master/Training/Damaged'):
#                print(roots,len(dirs),len(files))

#Dataset = '/home/nihar/Desktop/Pytorch-UNet-master/Training/Damaged'

class BasicDataset(Dataset):
    def __init__(self, imgs_dir,mask_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.ids = [];
        self.scale = scale
        
        # self.ids = [((root+'/'+files[0]),(root +'/'+ files[1])) for root,dirs,files in os.walk(imgs_dir)
        #             if len(files) > 1
        #             if (files[0].startswith('AP.jpg'))]
        #assert 0 < scale <= 1, 'Scale must be between 0 and 1'
	
                

        #self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #            if not file.startswith('.')]

        for dirpath,dirnames,filenames in os.walk('/content/drive/My Drive/TCDTIMIT/Noisy_TCDTIMIT/Cafe/20/volunteers'):
          print("present dir: ",dirpath)
          if dirpath.endswith("straightcam"):
            print(dirpath[67])
            if dirpath[67] == "0":

              for filename in filenames:
                if filename.endswith(".wav"):
                  print("wavfile: ",filename)
                  dirpath1 = '/content/drive/My Drive/TCDTIMIT/Clean/volunteers/' + dirpath[67:70] + '/straightcam'
                  temp = [(os.path.join(dirpath,filename)),(os.path.join(dirpath1,filename))]
                  self.ids.append(temp)
                
        
    def __len__(self):
        return len(self.ids)

    def preprocess(cls, pil_img, scale, NoiseCafe, NoiseCafe1):
        
                #print(Zxx.shape)
        for i in range(0,(T)*16,16):
          NoiseCafe.append(np.log(np.abs(Zxx[:,i:i+16])+1e-8))

        for i in range(0,(T)*16,16):
          NoiseCafe1.append(np.log(np.abs(Zxx[:,i:i+16])+1e-8))
        

        return (NoiseCafe, NoiseCafe1)

    def __getitem__(self, i):
        idxi = (self.ids[i])[0]
        idxm = (self.ids[i])[1]
        print(idxm)
        mask_file = glob(idxm)
        img_file = glob(idxi)
        
        # print("hi")
        NoiseCafe = []
        NoiseCafe1 = []

        
        
        img = (img_file[0])
        #print(img)
        sample_rate, samples = wavfile.read(img)
        f, t, Zxx = signal.stft(samples, sample_rate,nperseg=256,noverlap=192,nfft=256)
        T = len(t)//16

        for i in range(0,(T)*16,16):
          NoiseCafe.append(np.log(np.abs(Zxx[:,i:i+16])+1e-8))

        print(mask_file)
        mask = (idxm)
        sample_rate, samples = wavfile.read(mask)
        f, t, Zxx = signal.stft(samples, sample_rate,nperseg=256,noverlap=192,nfft=256)
        T = len(t)//16

        for i in range(0,(T)*16,16):
          NoiseCafe1.append(np.log(np.abs(Zxx[:,i:i+16])+1e-8))

        NoiseCafe = np.asarray(NoiseCafe)
        NoiseCafe1 = np.asarray(NoiseCafe1)

        return { "train" :NoiseCafe, "label" : NoiseCafe1}


#dataset = BasicDataset('hi', 'hi', 1)
#print(dataset.__getitem__(1))



   

