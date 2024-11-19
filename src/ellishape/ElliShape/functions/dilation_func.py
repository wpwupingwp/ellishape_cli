import cv2
import numpy as np

def dilation_func(imBina):
    # Define the structuring element
    B = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8)
    
    # Perform dilation
    im_dil = cv2.dilate(imBina.astype(np.uint8), B)
    
    return im_dil
