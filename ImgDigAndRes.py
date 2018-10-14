#图像的退化及其复原
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
import scipy.stats

def gaussieNoisy(img, mean=0,sigma=1):
    '''
    给图像加上高斯噪声,默认为标准高斯分布
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param mean: 高斯分布均值
    :param sigma: 高斯分布方差
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        height, width = img.shape
        guassNoise = np.random.normal(mean, sigma, size=(height, width))
    elif dimension == 3:
        height, width, channel = img.shape
        guassNoise = np.random.normal(mean, sigma, size=(height, width, channel))
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

    noisy = img + guassNoise
    return noisy.astype(np.uint8)

def rayleighNoisy(img,scale=1):
    '''
    给图像加上瑞利噪声
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param scale:
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        height, width = img.shape
        rayNoise = np.random.rayleigh(scale, size=(height, width))
    elif dimension == 3:
        height, width, channel = img.shape
        rayNoise = np.random.normal(scale, size=(height, width, channel))
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

    noisy = img + rayNoise
    return noisy.astype(np.uint8)

def pepperAndSalt(img, snr):
    '''
    椒盐噪声
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param snr: 信噪比
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        height, width = img.shape
        channel = 1;
    elif dimension == 3:
        height, width, channel = img.shape

    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))
    #根据信噪比计算要加噪的像素数目
    noiseNum = int((1-snr)*height*width*channel)
    result = np.copy(img)
    #在随机的位置将像素置为0或255
    for i in range(noiseNum):
        randX = int(np.random.random()*height)
        randY = int(np.random.random()*width)

        if np.random.randint(2)==0:
            result[randX, randY]=0
        else:
            result[randX, randY]=255
    return result

def img_meanFilter(img, size=(3,3)):
    #算术均值滤波器
    return cv.blur(img,size)

def geometricMeanOperator(roi):
    '''
    计算当前模板的算术均值滤波器
    :param roi:
    :return:
    '''
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return np.power(p, 1/(roi.shape[0]*roi.shape[1]))

def geometricMeanSingle(img):
    '''
    算术均值滤波器，对一个通道进行操作
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return:
    '''
    resultImg = np.copy(img)
    tempImg = cv.copyMakeBorder(img, 1,1,1,1,cv.BORDER_DEFAULT)
    for i in range(1, tempImg.shape[0]-1):
        for j in range(1, tempImg.shape[1]-1):
            resultImg[i-1, j-1] = geometricMeanOperator(tempImg[i-1:i+2, j-1:j+2])
    return resultImg.astype(np.uint8)

def img_gemotriccMean(img):
    '''
    算术均值滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return geometricMeanSingle(img)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = geometricMeanSingle(r)
        g = geometricMeanSingle(g)
        b = geometricMeanSingle(b)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

'''*****************谐波均值滤波器*****************'''
def HMeanOperator(roi):
    '''
    计算当前模板的算术均值滤波器
    :param roi:
    :return:
    '''
    roi = roi.astype(np.float64)
    if 0 in roi:
        return 0
    else:
        return scipy.stats.hmean(roi.reshape(-1))

def HMeanSingle(img, size = (3,3)):
    '''
    谐波均值滤波器，对一个通道进行操作
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :return:
    '''
    width = int(size[0]/2)
    height = int(size[1]/2)
    resultImg = np.copy(img)
    tempImg = cv.copyMakeBorder(img, height,height,width,width,cv.BORDER_DEFAULT)
    for i in range(height, tempImg.shape[0]-height):
        for j in range(width, tempImg.shape[1]-width):
            resultImg[i-height, j-width] = HMeanOperator(tempImg[i-height:i+height+1, j-width:j+width+1])
    return resultImg.astype(np.uint8)

def HMean(img, size=(3,3)):
    '''
    谐波均值滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param size: 滤波器大小
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return HMeanSingle(img, size)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = HMeanSingle(r, size)
        g = HMeanSingle(g, size)
        b = HMeanSingle(b, size)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))
'''***********************************************'''
'''*****************逆谐波均值滤波器*****************'''
def iHMeanOperator(roi, q):
    roi = roi.astype(np.float64)
    return np.mean(np.power(roi, q+1))/np.mean(np.power(roi, q))

def iHMeanSingle(img, q):
    '''
    逆谐波均值滤波器，对一个通道进行操作
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param q: 指数
    :return:
    '''
    resultImg = np.copy(img)
    tempImg = cv.copyMakeBorder(img, 1,1,1,1,cv.BORDER_DEFAULT)
    for i in range(1, tempImg.shape[0]-1):
        for j in range(1, tempImg.shape[1]-1):
            temp = tempImg[i-1:i+2, j-1:j+2]
            resultImg[i-1, j-1] = iHMeanOperator(temp,q)
    resultImg = (resultImg - np.min(img))*(255/np.max(img))
    return resultImg.astype(np.uint8)

def iHMean(img, q):
    '''
    逆谐波均值滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param q: 指数
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return iHMeanSingle(img,q)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = iHMeanSingle(r, q)
        g = iHMeanSingle(g, q)
        b = iHMeanSingle(b, q)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))
'''*************************************************'''
img = cv.imread('C:/Users/i349006/PycharmProjects/DIP3E_CH05_Original_Images/Fig0508(b)(circuit-board-salt-prob-pt1).tif', 0)
resultImg = iHMean(img,q=1.5)
cv.namedWindow('test')
cv.imshow('test', np.hstack([img,resultImg]))
cv.waitKey(0)

cv.destroyAllWindows()
