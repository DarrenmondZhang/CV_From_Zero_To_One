import cv2  # opencv读取的格式是BGR
import numpy as np
import matplotlib.pyplot as plt  # Matplotlib是RGB
from Digital_Image_Processing.basic_operation.basic_operation import *

img = cv2.imread('./pictures/cat.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape, img_gray.shape)

# cv_show('cat_gray', img_gray)


def shift_hsv(image):
    """BGR -> HSV"""
    image = cv_read(image)
    pic_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv_show('cat_hsv', pic_hsv)
    return pic_hsv


def threshold(image, operation=None):
    """
    阈值处理
    :param image: Original image input
    :param operation: Filter operation
    :return:  Processed image
    """
    image = cv_read(image)
    if operation == 'BINARY':
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    elif operation == 'BINARY_INV':
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    elif operation == 'TRUNC':
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    elif operation == 'TOZERO':
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    elif operation == 'TOZERO_INV':
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
    else:
        raise Exception('请输入正确的阈值处理方式')
    cv_show(operation, thresh)
    return thresh


def image_smoothing(image, kernel=None, operation=None, sigma=None):
    """
    图像平滑
    :param image: Original image input
    :param kernel: Convolution kernel size
    :param operation: Filter operation
    :param sigma: Gaussian kernel`s variance
    :return: Processed image
    """
    image = cv_read(image)
    # 均值滤波(简单的平均卷积操作),
    if operation == 'blur':
        image_result = cv2.blur(image, (kernel, kernel))
    elif operation == 'boxfilter':
        # 方框滤波: 基本和均值一样，可以选择归一化。不采取归一化的画，发生越界之后直接将值置为255
        image_result = cv2.boxFilter(image, -1, (kernel, kernel), normalize=True)
    elif operation == 'gaussian':
        # 高斯滤波:高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
        image_result = cv2.GaussianBlur(image, (kernel, kernel), sigma)
    elif operation == 'median':
        # 中值滤波: 相当于用中值代替
        image_result = cv2.medianBlur(image, kernel)  # 中值滤波
    cv_show(operation, image_result)
    return image_result


def morphology(image, kernel, iteration, type=None):
    """
    形态学处理
    :param image: Original image input
    :param kernel: Convolution kernel size
    :param iteration: Number of iterations when operation is dilation or erosion
    :param type: 形态学操作
    :return: Processed image
    """
    image = cv2.imread(image)
    if type == 'dilation':
        image_result = cv2.dilate(image, (kernel, kernel), iterations=iteration)
    elif type == 'erosion':
        image_result = cv2.erode(image, (kernel, kernel), iterations=iteration)
    elif type == 'open':
        # 先腐蚀 后膨胀
        image_result = cv2.morphologyEx(image, cv2.MORPH_OPEN, (kernel, kernel))
    elif type == 'close':
        # 先膨胀 后腐蚀
        image_result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (kernel, kernel))
    elif type == 'gradient':
        # 梯度运算 = 膨胀 - 腐蚀
        image_result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, (kernel, kernel))
    elif type == 'tophat':
        # 礼帽 = 原始输入 - 开运算结果
        image_result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, (kernel, kernel))
    elif type == 'blackhat':
        # 黑帽 = 闭运算 - 原始输入
        image_result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, (kernel, kernel))

    else:
        raise Exception("请输入正确的形态学操作")
    cv_show(type, image_result)
    return image_result


def gradient_operator(image, type=None):
    """
    梯度算子：sobel:一阶算子；laolacian:二阶算子
    :param image: Original image input
    :param type: Operator`s type
    :return: Processed image
    """
    image = cv2.imread(image)
    if type == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)  # 白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
        sobely = cv2.convertScaleAbs(sobely)
        image_result = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    elif type == 'scharr':
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        scharrx = cv2.convertScaleAbs(scharrx)
        scharry = cv2.convertScaleAbs(scharry)
        image_result = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    elif type == 'laplacian':
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        image_result = cv2.convertScaleAbs(laplacian)
    else:
        raise NameError("More operators are being added...")
    cv_show(type, image_result)
    return image_result


def canny_edge_detection(image, min_value, max_value):
    """
    边缘检测：canny
    :param image: Original image input
    :param min_value: Minimum threshold
    :param max_value: maximum threshold
    :return: Processed image
    """
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    image_result = cv2.Canny(image, min_value, max_value)
    cv_show('canny_result', image_result)
    return image_result


""" test code
shift_hsv('./pictures/cat.jpg')

threshold('./pictures/cat.jpg', operation='BINARY')

image_smoothing('./pictures/cat.jpg', kernel=3, operation='boxfilter', sigma=3)

morphology('./pictures/dige.png', 5, 3, type='open')

gradient_operator('./pictures/lena.jpg', type='sobel')

canny_edge_detection("./pictures/car.png", 120, 250)

"""
