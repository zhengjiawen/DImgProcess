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

img = cv.imread('C:/Users/i349006/PycharmProjects/DIP3E_CH05_Original_Images/Fig0503 (original_pattern).tif', 0)
resultImg = pepperAndSalt(img,0.6)
cv.namedWindow('test')
cv.imshow('test', np.hstack([img,resultImg]))
cv.waitKey(0)

cv.destroyAllWindows()
