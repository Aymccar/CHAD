import cv2 as cv
import numpy as np


def to_gray(frame) : 
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def sea_remove(frame) :
    frame_n = frame/np.percentile(frame, 75)
    mask = frame_n[:, :, 0].astype(bool)
    mask = (frame_n[:, :, 0] < 0.2) | (frame_n[:, :, 2] > 0.3)

    dilatation_size = 0
    dilation_shape = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(dilation_shape, (2*dilatation_size + 1, 2*dilatation_size +1), (dilatation_size, dilatation_size))
    erode_mask = cv.erode(mask.astype(np.uint8), element)
    
    return frame[:,:, 0].astype(bool)
    return erode_mask


def size_reduction(frame, n) :
    for i in range(n//2) :
        frame = cv.pyrDown(frame)
    return frame

def remove_part(frame) : 

    list_ = [[(0,250), (160, 650)]] 

    for rect in list_ :
        P1, P2 = rect[0], rect[1]
        frame[P1[0]:P1[1], P2[0]:P2[1], ...] = 0

    return frame

def clahe(frame) :
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame)


def mask(frame) :
    n = 2
    mask_ = size_reduction(frame, n)
    mask_ = sea_remove(mask_)
    mask_ = remove_part(mask_)

    frame = to_gray(frame)
    frame = size_reduction(frame, n)
    #frame = clahe(frame)

    return frame, mask_.astype(bool)


def mask_red(frame) : 
    n = 2
    mask_ = size_reduction(frame, n)
    mask_ = sea_remove(mask_)
    mask_ = remove_part(mask_)


    frame = sea_remove(frame)*frame[:,:, 2]
    frame = size_reduction(frame, n)
    frame = remove_part(frame)

    return frame, mask_.astype(bool)

    
