from Digital_Image_Processing.Image_Stich.Stitcher import Stitcher
from Digital_Image_Processing.basic_operation.basic_operation import *
import cv2

# 读取拼接图片
imageA = cv2.imread("left_01.png")
imageB = cv2.imread("right_01.png")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片
cv_show("Image A", imageA)
cv_show("Image B", imageB)
cv_show("Keypoint Matches", vis)
cv_show("Result", result)
