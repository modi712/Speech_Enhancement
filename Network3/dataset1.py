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

import stft

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

        specgram1 = stft.spectrogram(samples)

        frequencies, times, spectrogram1 = signal.spectrogram(samples, sample_rate)
        f, t, Zxx = signal.stft(samples, sample_rate,nperseg=256,noverlap=192,nfft=256)
        T = len(t)//16

        for i in range(0,(T)*16,16):
          NoiseCafe.append(np.log(np.abs(Zxx[:,i:i+16])+1e-8))

        

        X_train = NoiseCafe[0:8]
        Y_train = NoiseCafe[1:9]
        
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        Z_train = np.concatenate((X_train,Y_train))
        print(X_train.shape)
        print(Z_train.shape)
        for i in range(8,len(NoiseCafe)):
          Y_train = NoiseCafe[i-7:i+1]
          Y_train = np.asarray(Y_train)
          X_train = np.concatenate((X_train,Y_train))

        print(X_train.shape)
          #temp = NoiseCafe[i-7]
          #for j in range(1,8):
              #temp = np.concatenate((temp,NoiseCafe[(i-7)+j]),axis=1)
          #X_train = np.concatenate((X_train,temp))

        #NoiseCafe = X_train    

        print(mask_file)
        mask = (idxm)
        sample_rate, samples = wavfile.read(mask)

        specgram2 = stft.spectrogram(samples)
        frequencies, times, spectrogram2 = signal.spectrogram(samples, sample_rate)
        f, t, Zxx = signal.stft(samples, sample_rate,nperseg=256,noverlap=192,nfft=256)
        T = len(t)//16
        
        

        for i in range(0,(T)*16,16):
          NoiseCafe1.append(np.log(np.abs(Zxx[:,i:i+16])+1e-8))

        NoiseCafe = X_train
        NoiseCafe1 = np.asarray(NoiseCafe1)
        NoiseCafe1 = NoiseCafe1.reshape(NoiseCafe1.shape[0],1,129, 16)

        return { "image" :torch.from_numpy(spectrogram1), "mask" : torch.from_numpy(spectrogram2)}


#dataset = BasicDataset('hi', 'hi', 1)
#l = (dataset.__getitem__(1))
#x = l['image']
#y = l['mask']

#print(x.shape)
#print(y.shape)

   

