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

import cv2
import librosa
from pesq import pesq 

def train_net(net,
              device,
              epochs=20,
              batch_size=1,
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
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    epoch_loss = 0
    dice_epoch_loss = 0
    for epoch in range(20):
        net.train()
        print(epochs)
        print(epoch_loss)
        print(dice_epoch_loss)
        epoch_loss = 0
        dice_epoch_loss = 0;
        batch_temp = 0;
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                fs = batch['fs']
                a2 = batch['a2']
                a1 = batch['a1']
                print(batch_temp)
                batch_temp = batch_temp+1
                true_masks = batch['mask']
                #assert imgs.shape[1] == net.n_channels, \
                #    f'Network has been defined with {net.n_channels} input channels, ' \
                #    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                global_step += 1
                print(global_step, 'global_step')
                if global_step % 100 == 0:
                    #x = (masks_pred.cpu().data.numpy())
                    #x = x[0,:,:]
                    #x = x[0,:,:]
                    #x = (x > 0.5).astype(float)
                   
                    p = (true_masks.cpu().data.numpy())
                    a2 = (a2.cpu().data.numpy())
                    a1 = (a1.cpu().data.numpy())
                    #print(p.shape, 'h1')
                    p = p[0,0,:,:]
                    a2 = a2[0,:,:]
                    a1 = a1[0,:,:]
                    #print(p.shape, 'h2')
                    #p = p[0,:,:]
                    #p = (p > 0.5).astype(float)
                    #print(p)
                    #print(p.shape)
                    x = masks_pred
                    x = x[0,0,:,:]
                    x = (x.cpu().data.numpy())
                    #print(x, 'phappy')
                    #print(p, 'thappy')
                    q = x-p
                    q = np.abs(q)
                    print(q)
                    r = np.mean(q)
                    print(r, 'hey r')
                    print(np.mean(np.abs(x)), 'hey x')
                    print(np.mean(np.abs(p)), 'hey p')
                    y = str(global_step)
                    z = '/content/drive/My Drive/TCDTIMIT/img/apx'+y + '.png'
                    z1 = '/content/drive/My Drive/TCDTIMIT/img/atx'+y + '.png'
                    z2 = '/content/drive/My Drive/TCDTIMIT/audio/apwavx'+y + '.wav'
                    z3 = '/content/drive/My Drive/TCDTIMIT/audio/atwavx'+y + '.wav'
                    #print(z)
                    cv2.imwrite(z, x*255)
                    cv2.imwrite(z1, p*255)

                    #output = stft.ispectrogram(x)
                    #wav.write(z2, fs, output)

                    output = librosa.istft((np.sqrt(np.exp(p))) * (np.exp(1j*a2)))
                    librosa.output.write_wav(z3, output, fs)

                    output = librosa.istft(np.sqrt((np.exp(x))) * (np.exp(1j*a1)))
                    librosa.output.write_wav(z2, output, fs)

                    ref, rate = librosa.load(z3, sr=16000)
                    deg, rate = librosa.load(z2, sr=16000)

                    print('PESQ')
                    print(pesq(rate, ref, deg, 'wb'))

                #print(np.mean(masks_pred.cpu().data.numpy()))
                #pred = (masks_pred > 0.5).float()
                #dice_epoch_loss += dice_coeff(pred, true_masks.squeeze(dim=1)).item()
                #print(dice_coeff(pred, true_masks.squeeze(dim=1)).item())
                temp1 = net.inc.double_conv
                # print(temp1[0].weight)
                # m = nn.Sigmoid()
                loss = criterion((masks_pred), true_masks)
                print(loss,'- loss')
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                #global_step += 1
                if global_step % (100000) == 0:
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


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
