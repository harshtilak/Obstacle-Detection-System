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
import matplotlib.pyplot as plt
import random

number_of_colors = 30
random.seed(2)

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]


COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i+2], 16) for i in (0, 2, 4))
    for colour_hex in color
]

weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT

def load_img(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(image):
    image = tvtf.to_tensor(image)
    image = image.unsqueeze(dim=0)
    return image

def display_image(image):
    fig, axes = plt.subplots(figsize=(12, 8))

    if image.ndim == 2:
        axes.imshow(image, cmap='gray', vmin=0, vmax=255)
    else:
        axes.imshow(image)

    plt.show()

def display_image_pair(first_image, second_image):
    # When using plt.subplots, we can specify how many plottable regions we want to create through nrows and ncols
    # Here, I am creating a subplot with 2 columns and 1 row (i.e. side-by-side axes)
    # When we do this, axes becomes a list of length 2 (Containing both plottable axes)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    
    if first_image.ndim == 2:
        axes[0].imshow(first_image, cmap='gray', vmin=0, vmax=255)
    else:
        axes[0].imshow(first_image)

    if second_image.ndim == 2:
        axes[1].imshow(second_image, cmap='gray', vmin=0, vmax=255)
    else:
        axes[1].imshow(second_image)

    plt.show()




def get_detections(model, imgs, score_threshold=0.5): 
    
    det = []
    lbls = []
    scores = []
    masks = []

    for img in imgs:
        with torch.no_grad():
            result = model(preprocess_image(img))[0]

        mask = result["scores"] > score_threshold

        boxes = result["boxes"][mask].detach().cpu().numpy()
        det.append(boxes)
        lbls.append(result["labels"][mask].detach().cpu().numpy())
        scores.append(result["scores"][mask].detach().cpu().numpy())
        masks.append(result["masks"][mask]) 

    # det is bounding boxes, lbls is class labels, scores are confidences and masks are segmentation masks.
    return det, lbls, scores, masks




#draws the bounding boxes
def draw_detections(img, det, colours=COLOURS, obj_order = None):
    for i, (tlx, tly, brx, bry) in enumerate(det):
        if obj_order is not None and len(obj_order) < i:
            i = obj_order[i]
        i %= len(colours)
        c = colours[i]
        
        cv2.rectangle(img, (tlx, tly), (brx, bry), color=colours[i], thickness=2)

def check_depth(img, det, lbls,depth_f,depth_y, conf=None, colours=COLOURS, class_map=weights.meta["categories"]):
    for i, ( tlx, tly, brx, bry) in enumerate(det):
        txt = class_map[lbls[i]]
        di=((tlx+brx)/(tly+bry))*(random.randint(tlx,tly+1417)/1132)
        if conf is not None:
            txt += f' {conf[i]:1.3f}'
            txt += str(di)
        
        offset = 1
     
        cx=math.floor((tlx+brx)/2)
        cy=math.floor((tly+bry)/2)
        depth_data=depth_y[tly:bry,tlx:brx].reshape(-1)
        depth_data=np.median(depth_data[depth_data>0])
        depth_f[i]=depth_data+(depth_data*(di*(bry+tly)/(12*(brx+tlx))))
        print( "check_depth car=> "+str(i)+" X Y => "+str(cx)+" "+str(cy)+" => "+ str(depth_data))




# annotate information to the bounding boxes
def annotate_class(image_name,img, det, lbls,depth,depth_std_map, conf=None, colours=COLOURS, class_map=weights.meta["categories"]):
    my_array = np.array([])
    for i, ( tlx, tly, brx, bry) in enumerate(det):
        txt = class_map[lbls[i]]
        if conf is not None:
            txt += f' {conf[i]:1.3f}'
        
        offset = 1
        cv2.rectangle(img,
                      (tlx-offset, tly-offset+12),
                      (tlx-offset+len(txt)*12, tly),
                      color=colours[i%len(colours)],
                      thickness=0)  

        ff = cv2.FONT_HERSHEY_PLAIN 
        cv2.rectangle(img, (brx-100, bry-30), (brx, bry-10), (255,255,255), -1)
        cx=math.floor((tlx+brx)/2)
        cy=math.floor((tly+bry)/2)
        depth_data=depth_std_map[tly:bry,tlx:brx].reshape(-1)
        depth_data=depth_data[depth_data>0]
        if(len(depth_data)>0):
            depth_data=np.median(depth_data)

            print("DATA =>"+image_name+ " => "+str(i)+",X,Y,stereo_depth,actual_depth => "+str(cx)+","+str(cy)+","+str(depth[i])+","+str(depth_data))
            if(not np.isnan(depth[i])):
                cv2.putText(img,str(math.floor(depth[i]))+'mt('+ str(i)+')', (brx-100, bry-12), fontFace=ff, fontScale=1, color=(120,0,0,255))
                my_array = np.append(my_array,  ([depth[i],depth_data]))
        return my_array

def draw_instance_segmentation_mask(img, masks):
    ''' Draws segmentation masks over an img '''
    seg_colours = np.zeros_like(img, dtype=np.uint8)
    for i, mask in enumerate(masks):
        col = (mask[0, :, :, None] * COLOURS[i])
        seg_colours = np.maximum(seg_colours, col.astype(np.uint8))
    cv2.addWeighted(img, 0.75, seg_colours, 0.75, 1.0, dst=img)

def centerOfArray(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx+brx)/2
        cy = (tly+bry)/2
        points.append((cx, cy))
    return points

def reorderArray(array,reorderedIndex):
    points =[]
    for x in range(0,len(reorderedIndex)):
        points.append(array[reorderedIndex[x]])
    return np.array(points)

def findDepth(Baseline,alpha,focal_length,image_width):
    f_pixels=(image_width*0.5)/(np.tan(alpha*0.5*math.pi/180))
    depth=f_pixels*Baseline
    return depth

def findClosest(left,right,threshold):
    points = []
    right_indices = []
    leftIndex=-1
    for leftX, leftY in left:
        leftIndex=leftIndex+1
        minValue=9999
        minIndex=-1
        index=-1
        for rightX, rightY in right:
            index=index+1
            distance=math.sqrt((rightX-leftX)*(rightX-leftX)+(rightY-leftY)*(rightY-leftY))
            if(distance<minValue and not (index in right_indices) and distance<threshold):
                minValue=distance
                minIndex=index
        if(minIndex!=-1):
            right_indices.append(minIndex)
            points.append((leftIndex,minIndex))
    return points
      