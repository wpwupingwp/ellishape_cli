import cv2

def inverted_colors_func(imdata):
    inverted_imdata = cv2.bitwise_not(imdata)
    return inverted_imdata
