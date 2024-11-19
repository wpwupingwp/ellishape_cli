import cv2

def binarize_func(graydata):
    _, gray_bin = cv2.threshold(graydata, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray_bin
