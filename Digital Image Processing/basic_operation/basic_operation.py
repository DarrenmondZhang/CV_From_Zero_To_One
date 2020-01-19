import cv2  # openCV读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    """读取图片数据并显示"""
    cv2.imshow(winname=name, mat=img)
    cv2.waitKey(0)  # 设置等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()


def cv_save(name, img):
    """保存处理过后的图片"""
    cv2.imwrite(filename=name, img=img)


def cv_video(video):
    """读取视频"""
    vedio_ori = cv2.VideoCapture(video)
    # 检查打开是否正确
    if vedio_ori.isOpened():
        open, frame = vedio_ori.read()
    else:
        open = False
    while open:
        ret, frame = vedio_ori.read()
        if frame is None:
            break
        if ret:
            # 将彩色视频转换成灰度视频
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('result', gray)
            if cv2.waitKey(100) & 0xFF == 27:
                break
    vedio_ori.release()
    cv2.destroyAllWindows()


def cv_video1(video, operation):
    """读取视频"""
    vedio_ori = cv2.VideoCapture(video)
    # 检查打开是否正确
    if vedio_ori.isOpened():
        open, frame = vedio_ori.read()
    else:
        open = False
    while open:
        ret, frame = vedio_ori.read()
        if frame is None:
            break
        if ret:
            oper = cv2.cvtColor(frame, operation)

            cv2.imshow('result_grey', oper)
            if cv2.waitKey(100) & 0xFF == 27:
                break
    vedio_ori.release()
    cv2.destroyAllWindows()


gray = cv2.COLOR_BGR2GRAY
cv_video1('./pictures/test.mp4', operation=gray)

""" test code
cat_ori = cv2.imread('./pictures/cat.jpg')
cv_show('cat', cat_ori)

cat_grey = cv2.imread('./pictures/cat.jpg', cv2.IMREAD_GRAYSCALE)
cv_show('grey', cat_grey)

cv_save('./pictures/cat_grey.png', cat_grey)
cv_video('./pictures/test.mp4')

"""
