
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from data_generator import *
from metrics import dice_coef, dice_loss
from tensorflow.keras import backend as K

def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def DiceScore(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = tf.cast(K.flatten(inputs), tf.float32)
    targets = tf.cast(K.flatten(targets), tf.float32)
    
    intersection = K.sum(K.dot(tf.expand_dims(targets,0), tf.expand_dims(inputs,-1)))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return K.get_value(dice)

def IoUScore(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = tf.cast(K.flatten(inputs), tf.float32)
    targets = tf.cast(K.flatten(targets), tf.float32)
    
    intersection = K.sum(K.dot(tf.expand_dims(targets,0), tf.expand_dims(inputs,-1)))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return K.get_value(IoU)

def perf_measure(targets, inputs):
    #flatten label and prediction tensors
    y_hat = tf.cast(K.flatten(inputs), tf.bool)
    y_actual = tf.cast(K.flatten(targets), tf.bool)
    y_hat_not = tf.math.logical_not(y_hat)
    y_actual_not = tf.math.logical_not(y_actual)
    y_hat = tf.cast(y_hat, tf.float32)
    y_actual = tf.cast(y_actual, tf.float32)
    y_hat_not = tf.cast(y_hat_not, tf.float32)
    y_actual_not = tf.cast(y_actual_not, tf.float32)
    # TP = 0
    # FP = 0
    # TN = 0
    # FN = 0

    # for i in range(len(y_hat)): 
    #     if y_actual[i]==y_hat[i]==1:
    #        TP += 1
    #     if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
    #        FP += 1
    #     if y_actual[i]==y_hat[i]==0:
    #        TN += 1
    #     if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
    #        FN += 1

    TP = tf.keras.backend.get_value(K.sum(K.dot(tf.expand_dims(y_actual,0), tf.expand_dims(y_hat,-1))))
    FP = tf.keras.backend.get_value(K.sum(K.dot(tf.expand_dims(y_actual_not,0), tf.expand_dims(y_hat,-1))))
    TN = tf.keras.backend.get_value(K.sum(K.dot(tf.expand_dims(y_actual_not,0), tf.expand_dims(y_hat_not,-1))))
    FN = tf.keras.backend.get_value(K.sum(K.dot(tf.expand_dims(y_actual,0), tf.expand_dims(y_hat_not,-1))))  

    return(TP, FP, TN, FN)

def class_metrics(TP, FP, TN, FN):
  Precision = TP/(TP+FP)
  Recall = TP/(TP+FN)
  F1 = 2*Precision*Recall/(Precision+Recall)
  Specificity = TN/(TN+FP)
  Accuracy = (TP+TN)/(TP+TN+FP+FN)
  return(Precision, Recall, F1, Specificity, Accuracy)



if __name__ == "__main__":
    model_path = "files/resunetplusplus3_new.h5"
    save_path = "result"
    test_path = "data/val/"

    image_size = 256
    batch_size = 1

    test_image_paths = glob(os.path.join(test_path, "images", "*"))
    test_mask_paths = glob(os.path.join(test_path, "masks", "*"))
    test_image_paths.sort()
    test_mask_paths.sort()

    ## Create result folder
    try:
        os.mkdir(save_path)
    except:
        pass

    ## Model
    with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
        model = load_model(model_path)

    ## Test
    print("Test Result: ")
    test_steps = len(test_image_paths)//batch_size
    test_gen = DataGen(image_size, test_image_paths, test_mask_paths, batch_size=batch_size)
    model.evaluate_generator(test_gen, steps=test_steps, verbose=1)

    ## Generating the result
    Dice = []
    IOU = []
    Precision = []
    Recall = []
    F1 = []
    Specificity = []
    Accuracy = []
    for i, path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
        image = parse_image(test_image_paths[i], image_size)
        mask = parse_mask(test_mask_paths[i], image_size)

        predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
        predict_mask_ = (predict_mask > 0.5)
        predict_mask = predict_mask_ * 255.0
        
        Dice.append(DiceScore(mask,predict_mask_))
        IOU.append(IoUScore(mask,predict_mask_))
        TP, FP, TN, FN = perf_measure(mask,predict_mask_)
        Pr, Re, F1_, Sp, Acc = class_metrics(TP, FP, TN, FN)
        Precision.append(Pr)
        Recall.append(Re)
        F1.append(F1_)
        Specificity.append(Sp)
        Accuracy.append(Acc)

        sep_line = np.ones((image_size, 10, 3)) * 255

        mask = mask_to_3d(mask)
        predict_mask = mask_to_3d(predict_mask)

        all_images = [image * 255, sep_line, mask * 255, sep_line, predict_mask]
        cv2.imwrite(f"{save_path}/{test_image_paths[i].split('/')[-1][:-4]}.png", np.concatenate(all_images, axis=1))

    print("Test image generation complete")
    print("Average Test DICE score: ",sum(Dice)/len(Dice))
    print("Average Test IOU score: ",sum(IOU)/len(IOU))
    print("Average Test Precision score: ",sum(Precision)/len(Precision))
    print("Average Test Recall score: ",sum(Recall)/len(Recall))
    print("Average Test F1 score: ",sum(F1)/len(F1))
    print("Average Test Specificity score: ",sum(Specificity)/len(Specificity))
    print("Average Test Accuracy score: ",sum(Accuracy)/len(Accuracy))
