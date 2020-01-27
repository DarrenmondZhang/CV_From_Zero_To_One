import cv2  # openCV读取的格式是BGR
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


def pattern_matching(image, template, threshold, type=None):
    """
    多对象模板匹配
    :param image: 原始图片
    :param template: 模板图片
    :param threshold: 阈值
    :param type: 模板匹配算法
    :return:
    """
    image = cv2.imread(image)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template, 0)
    h, w = template.shape[:2]
    res = cv2.matchTemplate(image_grey, template, type)
    threshold = threshold
    # 取匹配程度大于阈值的坐标
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):  # *号表示可选参数
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(image, pt, bottom_right, (0, 0, 255), 2)

    cv_show('match_result', image)
    return image


def hist(image, operation=None):
    """
    直方图显示
    :param image: 原始图像
    :param operation:

    :return:
    """
    if operation == 'grey_show':
        image = cv2.imread(image, 0)
        plt.hist(image.ravel(), 256)
        plt.show()
    if operation == 'hist_show':
        image = cv2.imread(image)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
            plt.show()
    if operation == 'mask':
        image = cv2.imread(image, 0)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[100:300, 100:400] = 255
        masked_img = cv2.bitwise_and(img, img, mask=mask)  # 与操作
        hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
        hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.subplot(121), plt.imshow(masked_img, 'gray')
        plt.subplot(222), plt.plot(hist_full), plt.plot(hist_mask)
        plt.xlim([0, 256])
        plt.show()


def hist_equ(image, operation=None):
    """
    直方图均衡化
    :param image: 原始图像
    :param operation:
    :return:
    """
    image = cv2.imread(image, 0)  # 0表示灰度图 #clahe
    if operation == 'hist_equ':
        equ = cv2.equalizeHist(image)
    elif operation == 'adaptive_hist_equ':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equ = clahe.apply(image)
    cv_show('hist_equ', equ)
    return equ


def fourier_transform(image, type=None):
    """
    傅里叶变换
    :param image: 原始图像
    :param type: 高通滤波 vs. 低通滤波
    :return:
    """
    image = cv2.imread(image, 0)
    img_float32 = np.float32(image)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    if type == 'high_pass':
        # 低通滤波
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    elif type == 'low_pass':
        # 高通滤波
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    # IDFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('Result'), plt.xticks([]), plt.yticks([])

    plt.show()
    return img_back

fourier_transform('./pictures/lena.jpg', type='low_pass')
""" test code
shift_hsv('./pictures/cat.jpg')

threshold('./pictures/cat.jpg', operation='BINARY')

image_smoothing('./pictures/cat.jpg', kernel=3, operation='boxfilter', sigma=3)

morphology('./pictures/dige.png', 5, 3, type='open')

gradient_operator('./pictures/lena.jpg', type='sobel')

canny_edge_detection("./pictures/car.png", 120, 250)

pattern_matching('./pictures/mario.jpg', './pictures/mario_coin.jpg', 0.8, cv2.TM_CCOEFF_NORMED)

hist('./pictures/cat.jpg', 'mask')

hist_equ('./pictures/cat.jpg', operation='adaptive_hist_equ')

"""
