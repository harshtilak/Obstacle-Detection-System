import requests
import cv2
import numpy as np
import imutils
import os
import matplotlib.pyplot as plt

import copy
import math

import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
import torchvision
import torchvision.transforms.functional as tvtf
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights,MaskRCNN_ResNet50_FPN_V2_Weights
from pathlib import Path
from utils import *
from read_depth import *

my_array=np.array([])

def detect_depth(left_eye,right_eye,depth_filepath_left,hide_ui):
    global my_array
    depth_sh=depth_read(depth_filepath_left)
    
    left_img = load_img(left_eye)
    right_img = load_img(right_eye)
    
    image_width = right_img.shape[1]
    image_height_ = right_img.shape[0]
    if(not hide_ui):
        display_image_pair(left_img, right_img)

    imgs = [left_img, right_img]

    left_right = [preprocess_image(d).squeeze(dim=0) for d in imgs]

    print(right_img.shape)



    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    _ = model.eval()

    det, lbls, scores, masks = get_detections(model,imgs)

    reordered_right_array=findClosest(centerOfArray(det[0]),centerOfArray(det[1]),image_width/10)# to match the contours in both images
    left_indexes=[x[0] for x in reordered_right_array]
    right_indexes=[x[1] for x in reordered_right_array]

    det[0]=reorderArray(det[0],left_indexes)
    lbls[0]=reorderArray(lbls[0],left_indexes)

    

    det[1]=reorderArray(det[1],right_indexes)
    lbls[1]=reorderArray(lbls[1],right_indexes)
    

    det_left_final=centerOfArray(det[0])
    det_right_final=centerOfArray(det[1])
    det_left_final_X=[x[0] for x in det_left_final]
    det_right_final_X=[x[0] for x in det_right_final]

    disparity=abs(np.array(det_left_final_X)-np.array(det_right_final_X))
    focal_length=721.5377   
    alpha = 56.6 
    Baseline=54     
    print("Disparity objectwise: "+str(disparity))
    depth_of_objects=findDepth(Baseline,alpha,focal_length,image_width)/disparity


    for i, imgi in enumerate(imgs):
        img = imgi.copy()
        deti = det[i].astype(np.int32)
        check_depth(img,deti,lbls[i],depth_of_objects,depth_sh)
        break

    print("Depth of objects")
    print(depth_of_objects)
    print("left detections => ")
    print(np.array(weights.meta["categories"])[lbls[0]])
    print(det[0])
    print("right detections => ")
    print(np.array(weights.meta["categories"])[lbls[1]])
    print(det[1])

    if(not hide_ui):
        fig, axes = plt.subplots(2, 1, figsize=(15,15))


    for i, imgi in enumerate(imgs):
        img = imgi.copy()
        deti = det[i].astype(np.int32)
        print("Image No "+str(i))
        draw_detections(img,deti)
        
        my_array2=annotate_class(image_name,img,deti,lbls[i],depth_of_objects,depth_sh)
        print("my_array2")
        print(my_array2)
        my_array=np.append(my_array,my_array2)
        if(not hide_ui):
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Frame #{i}')
    if(not hide_ui):
        plt.show()


hide_ui=True
path = ""
for index in range(10,21):
    image_name="00000000"+str(index)+".png"
    stereo_image_dir="./dataset/stereo/"
    stereo_image_depth_dir="./dataset/depth/"
    left_eye_dir=stereo_image_dir+"image_02/data/"
    left_eye=left_eye_dir+image_name
    right_eye_dir=stereo_image_dir+"image_03/data/"
    right_eye=right_eye_dir+image_name
    depth_filepath_left=stereo_image_depth_dir+"image_02/"+image_name
    detect_depth(left_eye,right_eye,depth_filepath_left,hide_ui)
    print(my_array)
print(my_array)

stereo_depth = []
actual_depth = []
for i in range(0, len(my_array)):
    if i % 2:
        stereo_depth.append(my_array[i])
    else :
        actual_depth.append(my_array[i])

from sklearn.metrics import (
    mean_absolute_error, # MAE
    mean_squared_error # MSE,
    
)

MAE = mean_absolute_error(
    y_true=actual_depth, # actual values
    y_pred=stereo_depth # predicted values
)
MAE.round(2)

MSE = mean_squared_error(
    y_true=actual_depth, # actual values
    y_pred=stereo_depth # predicted values
)


print("RMSE -> "+str(math.sqrt(MSE.round(2))))
print("MSE -> "+str(MSE.round(2)))
print("MAE -> "+str(MAE.round(2)))