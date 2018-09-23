import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def imgGrayReverse(img):
    '''
    灰度图反色
    :param img:narray对象，灰度图是二维的
    :return: 反色后的narray对象
    '''
    height, width = img.shape
    temp = img.copy()

    for i in range(height):
        for j in range(width):
            temp[i, j] = 255 - img[i, j]

    return temp

def imgColorReverse(img):
    '''
    灰度图反色
    :param img:narray对象，RGB图是三维的
    :return: 反色后的narray对象
    '''
    height, width ,channel= img.shape
    temp = img.copy()

    for i in range(height):
        for j in range(width):
            temp[i, j] = 255 - img[i, j]

    return temp

def imgReverse(img):
    '''
    灰度图反色
    :param img:narray对象，灰度图是二维的,RGB图是三维的
    :return: 反色后的narray对象
    '''
    dimension = len(img.shape)
    if dimension == 2:
        height, width = img.shape
    elif dimension == 3:
        height, width, channel = img.shape
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

    temp = img.copy()

    for i in range(height):
        for j in range(width):
            temp[i, j] = 255 - img[i, j]

    return temp

def imgLogProcess(img, c=1):
    '''
    对数变换
    :param img: narray对象，灰度图是二维的,RGB图是三维的
            c:  常数
    :return: log变换后的narray对象
    '''
    dimension = len(img.shape)
    if dimension == 2:
        height, width = img.shape
    elif dimension == 3:
        height, width, channel = img.shape
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))
    temp = img.copy()

    for i in range(height):
        for j in range(width):
            temp[i, j] = c*np.log(1+img[i, j])

    return temp

def imgGamaProcess(img, c=1, gama=1):
    '''
    伽马变换
    :param img: narray对象，灰度图是二维的,RGB图是三维的
            c:  常数
    :return: log变换后的narray对象
    '''
    dimension = len(img.shape)
    if dimension == 2:
        height, width = img.shape
    elif dimension == 3:
        height, width, channel = img.shape
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))
    temp = img.copy()

    for i in range(height):
        for j in range(width):
            temp[i, j] = c*np.power(img[i, j], gama)

    return temp

def showHist(img):
    '''
    展示直方图
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return: 
    '''
    dimension = len(img.shape)
    if dimension == 2:
        showGrayHist(img)
    elif dimension == 3:
        showColorHist(img)
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

def showGrayHist(img):
    '''
    展示灰度图的直方图
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return: 
    '''
    hist = cv.calcHist([img], [0], None, [256], [0, 255])
    plt.figure()
    plt.title("GrayScale Hist")
    plt.xlabel("Bins")
    plt.ylabel("num of Pixels")
    plt.plot(hist)
    plt.xlim([0, 255])
    plt.show()

def showColorHist(img):
    '''
    展示RGB图的直方图
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return: 
    '''

    channels = cv.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("ColorScale Hist")
    plt.xlabel("Bins")
    plt.ylabel("num of Pixels")
    for (channel, color) in zip(channels, colors):
        hist = cv.calcHist([channel], [0], None, [256], [0, 255])
        plt.plot(hist, color = color)
        plt.xlim([0, 255])
    plt.show()

def imgEqualizeHist(img):
    '''
    直接用opencv的equalizeHist方法进行直方图均衡化，增加了对RGB图的逻辑
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return: 
    '''
    dimension = len(img.shape)
    if dimension == 2:
        #灰度图直接使用opencv的方法
        eq = cv.equalizeHist(img)
    elif dimension == 3:
        #先转成YUV通道，然后对Y通道进行均衡化，再转回RGB通道
        img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv.equalizeHist(img[:,:,0])
        eq = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))
    return eq

img = cv.imread("F:/lena.jpg",1)
eq = imgEqualizeHist(img)

cv.imshow("Histgram Equalization", np.hstack([img,eq]))
cv.waitKey(0)
showHist(img)
showHist(eq)


# test = imgGamaProcess(img,10,0.2)
# cv.namedWindow("test")
# cv.imshow("test", test)
# cv.waitKey(0)
# cv.destroyAllWindows()

