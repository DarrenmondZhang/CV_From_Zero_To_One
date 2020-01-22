### 01. HSV
- H - 色调（主波长）。 
- S - 饱和度（纯度/颜色的阴影）。 
- V值（强度）

### 02. 图像阈值
```python
ret, dst = cv2.threshold(src, thresh, maxval, type)
```
- src： 输入图，只能输入单通道图像，通常来说为灰度图
- dst： 输出图
- thresh： 阈值
- maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
- type：二值化操作的类型，包含以下5种类型： 
    - cv2.THRESH_BINARY           超过阈值部分取maxval（最大值），否则取0
    - cv2.THRESH_BINARY_INV    THRESH_BINARY的反转
    - cv2.THRESH_TRUNC            大于阈值部分设为阈值，否则不变
    - cv2.THRESH_TOZERO          大于阈值部分不改变，否则设为0
    - cv2.THRESH_TOZERO_INV  THRESH_TOZERO的反转
    
### 03. 图像平滑
* 均值滤波


### 04. Canny边缘检测
1. 使用高斯滤波器，以平滑图像，滤除噪声。
2. 计算图像中每个像素点的梯度强度和方向。
3. 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
4. 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
5. 通过抑制孤立的弱边缘最终完成边缘检测。

#### 4.1 高斯滤波器
![title](./pictures/canny_1.png)
#### 4.2 计算梯度和方向（用sobel算子）
![title](./pictures/canny_2.png)
#### 4.3 非极大值抑制
![title](./pictures/canny_3.png)
![title](./pictures/canny_6.png)
#### 4.4 双阈值检测
![title](./pictures/canny_5.png)

### 05. 图像金字塔
- 高斯金字塔
- 拉普拉斯金字塔

![title](./pictures/Pyramid_1.png)
 
#### 5.1 高斯金字塔

* 向下采样方法（缩小）
![title](./pictures/Pyramid_2.png)

* 向上采样方法（放大）
![title](./pictures/Pyramid_3.png)

#### 5.2 拉普拉斯金字塔
![title](./pictures/Pyramid_4.png)

### 06. 图像轮廓
```python
cv2.findContours(img,mode,method)
```
mode:轮廓检索模式
- RETR_EXTERNAL ：只检索最外面的轮廓；
- RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
- RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
- RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;

method:轮廓逼近方法
- CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
- CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。