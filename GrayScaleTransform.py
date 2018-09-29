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

def imgEqualizeHistByLut(img):
    '''
    使用查找表的方式，手动实现直方图均衡化
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return: 
    '''
#    lut = np.zeros(256, dtype = img.dtype)
    hist, bins = np.histogram(img.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)  # 除去直方图中的0值
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式
    cdf = np.ma.filled(cdf_m,0).astype('uint8') #将掩模处理掉的元素补为0
    #lut查表操作，根据表中元素映射灰度值
    return cv.LUT(img, cdf)

def filterProcess(img, kernel):
    #自定义卷积操作
    return cv.filter2D(img, -1, kernel)

def laplaceProcess(img, ksize=1, scale=1, delta=0, borderType=cv.BORDER_DEFAULT):
    #拉普拉斯算子进行滤波
    dst = cv.Laplacian(img,-1, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
    return dst

def sobelProcess(img, dx, dy, ksize=1, scale=1,delta=0, borderType=cv.BORDER_DEFAULT):
    #sobel算子进行滤波
    dst = cv.Sobel(img,-1, dx,dy,ksize=ksize, scale=scale, delta=delta, borderType=borderType)
    return dst

#骨骼增强图像实验代码

sobelXKernel = np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]])
sobelYKernel = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]])
laplaceKernel = np.array([[-1,-1,-1],
                          [-1,8,-1],
                          [-1,-1,-1]])
avgKernel = np.ones((5,5))/25


img = cv.imread("F:/testImg/DIP3E_Original_Images_CH03/Fig0343(a)(skeleton_orig).tif",0)
#laplaceResult = laplaceProcess(img, ksize=3)
laplaceResult = filterProcess(img, laplaceKernel)
np.where(laplaceResult>0,laplaceResult,0)

#生成sobel图像
# sobelResultX = sobelProcess(img,1,0,ksize=3)
# sobelResultY = sobelProcess(img,0,1,ksize=3)
sobelResultX = filterProcess(img,sobelXKernel)
sobelResultY = filterProcess(img,sobelYKernel)
#sobelResult = np.square(np.power(sobelResultX,2)+np.power(sobelResultY,2))
sobelResult = np.abs(sobelResultX)+np.abs(sobelResultY)

img_blur = cv.blur(sobelResult,ksize=(5,5))
#img_blur = filterProcess(sobelResult, avgKernel)
result = np.multiply(laplaceResult+img,sobelResult)

#cv.imshow("test", np.hstack([img,result+img]))
cv.imshow("test", result)
cv.waitKey(0)


# test = imgGamaProcess(img,10,0.2)
# cv.namedWindow("test")
# cv.imshow("test", test)
# cv.waitKey(0)
# cv.destroyAllWindows()

