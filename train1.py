import pickle as pk
import tensorflow as tf
import numpy as np
import os

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from dataset1 import BasicDataset
from torch.utils.data import DataLoader, random_split
from dice_loss import dice_coeff
dir_img = '/content/drive/My Drive/ML /Training/Damaged'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'



import keras
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose,Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
    

model = Sequential()
model.add(Conv2D(64, kernel_size=(7, 7),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',data_format= "channels_first",input_shape=(8,129,16)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (5, 5),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',data_format= "channels_first"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',data_format= "channels_first"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (1, 1),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',data_format= "channels_first"))
#model.add(Conv2DTranspose(128, (3, 3), activation='relu',padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2DTranspose(128, (3, 3),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',data_format= "channels_first"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2DTranspose(64, (5, 5),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',data_format= "channels_first"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2DTranspose(1, (7, 7),padding='valid',use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros',data_format= "channels_first"))
model.add(BatchNormalization())
model.add(Activation('relu'))


def train_net(net,
              device,
              epochs=20,
              batch_size=16,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    lr = 0.011
    momentum = 0.99
    weight_decay = 1e-6
    lr_decay = 0.01
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    print(len(dataset))
    # for i in range(len(dataset.ids)):
    #   dataset.__getitem__(i)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)
    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay,lr_decay = lr_decay)
    
        
    
    criterion = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
    epoch_loss = 0
    dice_epoch_loss = 0
    for epoch in range(20):
        #net.train()
        print(epochs)
        print(epoch_loss)
        print(dice_epoch_loss)
        epoch_loss = 0
        dice_epoch_loss = 0;
        batch_temp = 0;
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                x_train = batch['train']
                print(batch_temp)
                batch_temp = batch_temp+1;
                y_train = batch['label']
                #assert imgs.shape[1] == net.n_channels, \
                #    f'Network has been defined with {net.n_channels} input channels, ' \
                #    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #    'the images are loaded correctly.'

                x_train = x_train.to(device=device)
                 
                y_train = y_train.to(device=device)

                y_pred = model(x_train)
                print(np.mean(y_train.cpu().data.numpy()))
                #pred = (masks_pred > 0.5).float()
                #dice_epoch_loss += dice_coeff(pred, true_masks.squeeze(dim=1)).item()
                #print(dice_coeff(pred, true_masks.squeeze(dim=1)).item())
                #temp1 = net.inc.double_conv
                # print(temp1[0].weight)
                # m = nn.Sigmoid()
                loss = criterion(y_pred, y_train)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = 0
                    for batch in val_loader:
                      val_score = val_score + eval_net(net, batch, device, 0)/n_val
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    # val_score = eval_net(net, val_loader, device, n_val)
    # if net.n_classes > 1:
    #     logging.info('Validation cross entropy: {}'.format(val_score))
    #     writer.add_scalar('Loss/test', val_score, global_step)

    # else:
    #     logging.info('Validation Dice Coeff: {}'.format(val_score))
    #     writer.add_scalar('Dice/test', val_score, global_step)

    #     writer.add_images('images', imgs, global_step)
    # if net.n_classes == 1:
    #     writer.add_images('masks/true', true_masks, global_step)
    #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
    writer.close()

