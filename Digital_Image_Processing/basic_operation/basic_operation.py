import cv2  # openCV读取的格式是BGR


def cv_read(img):
    """读取图片"""
    img = cv2.imread(img)
    return img


def cv_show(name, img):
    """显示图片"""
    cv2.imshow(winname=name, mat=img)
    cv2.waitKey(0)  # 设置等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()


def cv_save(name, img):
    """保存处理过后的图片"""
    cv2.imwrite(filename=name, img=img)


def cv_video(video, operation):
    """读取视频"""
    video_ori = cv2.VideoCapture(video)
    # 检查打开是否正确
    if video_ori.isOpened():
        open, frame = video_ori.read()
    else:
        open = False
    while open:
        ret, frame = video_ori.read()
        if frame is None:
            break
        if ret:
            oper = cv2.cvtColor(frame, operation)

            cv2.imshow('result_grey', oper)
            if cv2.waitKey(100) & 0xFF == 27:
                break
    video_ori.release()
    cv2.destroyAllWindows()


def roi_crop(img, w_l, w_r, h_l, h_r, img_roi_name):
    """截取部分图像数据 ROI"""
    img = cv2.imread(img)
    img_roi = img[w_l:w_r, h_l:h_r]
    cv_show(img_roi_name, img_roi)


def channels_split(img, channel):
    """通道拆分"""
    img = cv2.imread(img)
    # b, g, r = cv2.split(img)
    if channel == 'r':
        img_split = img.copy()
        img_split[:, :, 0] = 0
        img_split[:, :, 1] = 0
        cv_show('R', img_split)
    elif channel == 'g':
        img_split = img.copy()
        img_split[:, :, 0] = 0
        img_split[:, :, 2] = 0
        cv_show('G', img_split)
    elif channel == 'b':
        img_split = img.copy()
        img_split[:, :, 1] = 0
        img_split[:, :, 2] = 0
        cv_show('B', img_split)
    else:
        print("请输入正确的通道")


def makeborder(img, top_size, bottom_size, left_size, right_size, option=None):
    """
    - BORDER_REPLICATE：复制法，也就是复制最边缘像素。
    - BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
    - BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
    - BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
    - BORDER_CONSTANT：常量法，常数值填充。
    """
    ing_ori = cv2.imread(img)
    if option == 'replicate':
        replicate = cv2.copyMakeBorder(ing_ori, top_size, bottom_size, left_size, right_size,
                                       borderType=cv2.BORDER_REFLECT)
        cv_show('replicate_border', replicate)
    elif option == 'reflect':
        reflect = cv2.copyMakeBorder(ing_ori, top_size, bottom_size, left_size, right_size,
                                     borderType=cv2.BORDER_REFLECT)
        cv_show('reflect_border', reflect)
    elif option == 'reflect101':
        reflect_101 = cv2.copyMakeBorder(ing_ori, top_size, bottom_size, left_size, right_size,
                                         borderType=cv2.BORDER_REFLECT_101)
        cv_show('reflect101_border', reflect_101)
    elif option == 'wrap':
        wrap = cv2.copyMakeBorder(ing_ori, top_size, bottom_size, left_size, right_size,
                                  borderType=cv2.BORDER_WRAP)
        cv_show('wrap_border', wrap)
    elif option == 'constant':
        constant = cv2.copyMakeBorder(ing_ori, top_size, bottom_size, left_size, right_size,
                                      borderType=cv2.BORDER_CONSTANT, value=0)
        cv_show('constant_border', constant)


def cv_fusion(img1, img2, rate1, rate2, size):
    """图像融合，前提:两张图片尺寸相同"""
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    fusion_result = cv2.addWeighted(img1, rate1, img2, rate2, 0)
    cv_show('fusion_result', fusion_result)


""" test code
cat_ori = cv2.imread('./pictures/cat.jpg')
cv_show('cat', cat_ori)

cat_grey = cv2.imread('./pictures/cat.jpg', cv2.IMREAD_GRAYSCALE)
cv_show('grey', cat_grey)

cv_save('./pictures/cat_grey.png', cat_grey)
cv_video('./pictures/test.mp4')

# 对视频进行操作
gray = cv2.COLOR_BGR2GRAY
cv_video1('./pictures/test.mp4', operation=gray)

# 图像截取 ROI
roi_crop('./pictures/cat.jpg', 20, 100, 30, 200, 'cat_roi')

# 通道拆分
channels_split('./pictures/cat.jpg', 'g')

# 边界填充
makeborder('./pictures/cat.jpg', 50, 50, 50, 50, 'constant')

# 图像融合
cv_fusion('./pictures/cat.jpg', './pictures/dog.jpg', 0.2, 0.8, size=(500, 414))


"""
