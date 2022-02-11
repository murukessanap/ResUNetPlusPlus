import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

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


smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
