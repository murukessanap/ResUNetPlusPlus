import argparse
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from data_generator import DataGen
from unet import Unet
from resunet import ResUnet
from m_resunet import ResUnetPlusPlus
from metrics import dice_coef, dice_loss

import keras
import keras.backend as K


def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    # print(inputs.shape)
    # print(targets.shape)
    
    intersection = K.sum(targets*inputs)
    dice = (2.*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def IoULoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets*inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (1.*intersection + smooth) / (union + smooth)
    return 1 - IoU


ALPHA = 0.5
BETA = 0.5
def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
        
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    
    return 1 - Tversky

GAMMA = 0.5
def SSLoss(targets, inputs, gamma=GAMMA, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    sq = K.square(targets-inputs)
    inputs_o = 1 - inputs
    LSS = gamma*(K.sum(sq*inputs)+smooth)/(K.sum(inputs)+smooth) + (1-gamma)*(K.sum(sq*inputs_o)+smooth)/(K.sum(inputs_o)+smooth)
    
    return LSS

loss_fn_dict = {"DiceLoss":DiceLoss,"IoULoss":IoULoss,"TverskyLoss":TverskyLoss,"SSLoss":SSLoss,"binary_crossentropy":"binary_crossentropy"}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ResUnetPlusPlus argument list")
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--lr", help="learning rate")
    parser.add_argument("--epochs", help="no of epochs")
    parser.add_argument("--loss_fn", help="loss function to use in training")

    args = parser.parse_args()
    
    ## Path
    file_path = "files/"
    model_path = "files/resunetplusplus3_new.h5"

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path = "data/train/"
    valid_path = "data/val/"

    ## Training
    train_image_paths = glob(os.path.join(train_path, "images", "*"))
    train_mask_paths = glob(os.path.join(train_path, "masks", "*"))
    train_image_paths.sort()
    train_mask_paths.sort()

    # train_image_paths = train_image_paths[:2000]
    # train_mask_paths = train_mask_paths[:2000]

    ## Validation
    valid_image_paths = glob(os.path.join(valid_path, "images", "*"))
    valid_mask_paths = glob(os.path.join(valid_path, "masks", "*"))
    valid_image_paths.sort()
    valid_mask_paths.sort()

    ## Parameters
    image_size = 256
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    epochs = int(args.epochs)
    loss_fn = loss_fn_dict[args.loss_fn]
    
    #batch_size = 8
    #lr = 1e-4
    #lr = 1e-3
    #lr = 3e-2
    #lr = 1e-5
    #epochs = 200

    train_steps = len(train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size
    
    #train_steps = 1
    #valid_steps = 1

    ## Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    ## Unet
    #arch = Unet(input_size=image_size)
    #model = arch.build_model()

    ## ResUnet
    #arch = ResUnet(input_size=image_size)
    #model = arch.build_model()

    ## ResUnet++
    arch = ResUnetPlusPlus(input_size=image_size)
    model = arch.build_model()

    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
    #model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

    csv_logger = CSVLogger(f"{file_path}resunet3_{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_dice_coef', mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-8, verbose=1)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    #callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]
    callbacks = [csv_logger, checkpoint, reduce_lr]
    #callbacks = [csv_logger, reduce_lr]

    model.fit_generator(train_gen,
            validation_data=valid_gen,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            epochs=epochs,
            callbacks=callbacks)
