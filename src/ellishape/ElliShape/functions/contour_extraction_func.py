import cv2
import numpy as np
from scipy.signal import convolve
def contour_extraction_func(graydata,mode,threshvalue1,threshvalue2):
    # Apply edge detection
    if mode==1:
        im_contour = cv2.Canny(graydata, threshvalue1,threshvalue2)  #canny
    elif mode==2: #sobel
        sobelx = cv2.Sobel(graydata, cv2.CV_64F, 1, 0, ksize=15)
        sobely = cv2.Sobel(graydata, cv2.CV_64F, 0, 1, ksize=15)
        im_contour = cv2.magnitude(sobelx, sobely)
    elif mode==3:   #zerocross
        # 定义一个零交叉核
        kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

        # 使用 convolve 函数应用零交叉核
        filtered_image = convolve(graydata, kernel)
        # 生成二值图像，将零交叉点设为边缘点
        im_contour = np.zeros_like(filtered_image)
        im_contour[filtered_image > 0] = 255
    elif mode==4:#laplace
        im_contour=cv2.Laplacian(graydata, cv2.CV_64F)
    elif mode==5:
            # 使用 Roberts 算子进行边缘检测
        roberts_kernel_x = np.array([[1, 0], [0, -1]])
        roberts_kernel_y = np.array([[0, 1], [-1, 0]])
        # 对图像进行卷积
        gradient_x = cv2.filter2D(graydata, -1, roberts_kernel_x)
        gradient_y = cv2.filter2D(graydata, -1, roberts_kernel_y)
        # 计算梯度幅值
        magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        # 将梯度幅值映射到 0-255 的范围
        magnitude = (magnitude / np.max(magnitude)) * 255
        im_contour = magnitude.astype(np.uint8)
    elif mode==6:
            # 使用 Prewitt 算子进行边缘检测
        prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # 对图像进行卷积
        gradient_x = cv2.filter2D(graydata, -1, prewitt_kernel_x)
        gradient_y = cv2.filter2D(graydata, -1, prewitt_kernel_y)

        # 计算梯度幅值
        magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

        # 将梯度幅值映射到 0-255 的范围
        magnitude = (magnitude / np.max(magnitude)) * 255
        im_contour = magnitude.astype(np.uint8)
    
    return im_contour



