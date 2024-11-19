import cv2

def grayscale_func(imdata):
    graydata = cv2.cvtColor(imdata, cv2.COLOR_BGR2GRAY)
    return graydata
