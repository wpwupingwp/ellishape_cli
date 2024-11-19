import cv2

def enhancement_func(graydata):
    # Perform contrast and brightness adjustment
    gray_en = cv2.convertScaleAbs(graydata, alpha=1.5, beta=50)
    
    return gray_en
