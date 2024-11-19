import cv2
import numpy as np

def corrosion_func(imBina, circle):
    # Create a flat disk structuring element with the specified radius
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * circle + 1, 2 * circle + 1))
    
    # Perform erosion
    im_cor = cv2.erode(imBina.astype(np.uint8), se1)
    
    return im_cor
