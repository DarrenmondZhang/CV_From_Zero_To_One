import cv2
import random
import numpy as np


img_ori = cv2.imread('lenna.jpg', 1)
# img_ori = cv2.imread('dark.jpg', 1)
img_grey = cv2.imread('lenna.jpg', 0)
print(img_ori.shape)
print(img_ori.shape[0])
print(img_ori.shape[1])


def image_show(img):
    cv2.imshow('lenna', img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


def image_crop(h1, h2, w1, w2):
    if h1 < 0 or h2 < 0 or w1 < 0 or w2 < 0 \
            or h1 > img_ori.shape[0] or h2 > img_ori.shape[0] \
            or w1 > img_ori.shape[1] or w2 > img_ori.shape[1]:
        print('你输入有误，请重新输入')
    else:
        img_crop = img_ori[h1:h2, w1:w2]
        image_show(img_crop)


def color_shift(img, b_rand, g_rand, r_rand):
    # brightness
    B, G, R = cv2.split(img)

    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    image_show(img_merge)


def adjust_gamma(img, gamma=1.0):
    invgamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i/255.0) ** invgamma) * 255)
    table = np.array(table).astype('uint8')
    image_dark = cv2.LUT(img, table)
    image_show(image_dark)


def rotation(img):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1)  # center, angle, scale
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    image_show(img_rotate)


def affine_transform(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    image_aff = cv2.warpAffine(img, M, (cols, rows))
    image_show(image_aff)


def bright_img():
    """YUV色彩空间的Y进行直方图均衡来调节亮度"""
    imgyuv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YUV)
    imgyuv[:, :, 0] = cv2.equalizeHist(imgyuv[:, :, 0])
    imgout = cv2.cvtColor(imgyuv, cv2.COLOR_YUV2BGR)
    image_show(imgout)


def perspective_transform():
   pass


if __name__ == '__main__':
    image_show(img_ori)
    image_crop(150, 300, 100, 300)
    # color_shift(img_ori, b_rand=100, g_rand=40, r_rand=150)
    # rotation(img_ori)
    # adjust_gamma(img_ori, 2)
    # affine_transform(img_ori)
    # bright_img()