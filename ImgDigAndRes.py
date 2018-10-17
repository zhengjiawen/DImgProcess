#图像的退化及其复原
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from ImgDigAndResConstants import const

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
    计算当前模板的几何均值滤波器
    :param roi:
    :return:
    '''
    roi = roi.astype(np.float64)
    p = np.prod(roi)
    return np.power(p, 1/(roi.shape[0]*roi.shape[1]))

def geometricMeanSingle(img):
    '''
    几何均值滤波器，对一个通道进行操作
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
    几何均值滤波器
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
    width = int(size[1]/2)
    height = int(size[0]/2)
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
    return np.mean((roi)**(q+1))/np.mean((roi)**(q))

def iHMeanSingle(img, q):
    '''
    逆谐波均值滤波器，对一个通道进行操作
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param q: 指数
    :return:
    '''
    resultImg = np.zeros(img.shape)
    tempImg = cv.copyMakeBorder(img, 1,1,1,1,cv.BORDER_DEFAULT)
    for i in range(1, tempImg.shape[0]-1):
        for j in range(1, tempImg.shape[1]-1):
            temp = tempImg[i-1:i+2, j-1:j+2]

            resultImg[i - 1, j - 1] = iHMeanOperator(temp,q)

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

'''*************************统计排序滤波器************************'''
def madianFilter(img, size=3):
    '''
    中值滤波器
    :param img:
    :param size:
    :return:
    '''
    return cv.medianBlur(img, size)


def maxFilterOperator(roi):
    '''
    最大值滤波器，对模板内的部分进行计算
    :param roi: 部分图像矩阵
    :return:
    '''
    return np.max(roi)

def maxFilterSingle(img, size=(3,3)):
    '''
    最大值滤波器，单通道
    :param roi:
    :param size: rows,cols
    :return:
    '''
    width = int(size[1]/2)
    height = int(size[0]/2)
    resultImg = np.zeros(img.shape)
    tempImg = cv.copyMakeBorder(img, height,height,width,width,cv.BORDER_DEFAULT)
    for i in range(height, tempImg.shape[0]-height):
        for j in range(width, tempImg.shape[1]-width):
            resultImg[i-height, j-width] = maxFilterOperator(tempImg[i-height:i+height+1, j-width:j+width+1])
    return resultImg.astype(np.uint8)

def maxFilter(img, size=(3,3)):
    '''
    最大值滤波器
    :param img:
    :param size:
    :return:
    '''
    if size[0]%2 != 0 and size[1]%2 != 0and size[0] == 0 and size[1] == 0:
        raise (RuntimeError("滤波器大小必须为奇数"))
    dimension = len(img.shape)
    if dimension == 2:
        return maxFilterSingle(img,size)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = maxFilterSingle(r, size)
        g = maxFilterSingle(g, size)
        b = maxFilterSingle(b, size)
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

def minFilterOperator(roi):
    '''
    最小值滤波器，对模板内的部分进行计算
    :param roi: 部分图像矩阵
    :return:
    '''
    return np.min(roi)

def medianFilterOperator(roi):
    '''
    中值滤波器
    :param roi:
    :return:
    '''
    tempArray = roi.flatten()
    size = len(tempArray)
    tempArray.sort()
    return tempArray[int(size/2)]

def minpointFilterOperator(roi):
    return (int(np.max(roi))+int(np.min(roi)))/2

def alphaMeanFilterAmendOperator(roi, d):
    '''
    修正的aloha均值滤波器
    :param roi:
    :param d:
    :return:
    '''
    tempArray = roi.flatten()
    size = len(tempArray)
    if 2*d >= size:
        raise (RuntimeError("d不能超过滤波器元素的一半"))
    tempArray.sort()
    alphaAmendArray = tempArray[int(d/2):int(size-d/2)]
    return np.mean(alphaAmendArray)
'''由于前面重复代码过多，这里做个整合，将滤波器都整合到一起'''
def imgFilter(img, size=3,filterType=const.MAX_FILTER, q=1, d=2):
    '''
    滤波器
    :param img: narray对象，灰度图是二维的,RGB图是三维的
    :param size: 滤波器大小，size=（rows,cols）
    :param filterType: 滤波器类型，在constants类中有定义
    :return:
    '''
    if size%2 != 0 and size == 0 :
        raise (RuntimeError("滤波器大小必须为奇数"))
    dimension = len(img.shape)
    if dimension == 2:
        return imgFilterSingle(img,size, filterType,q,d)
    elif dimension == 3:
        r, g, b = cv.split(img)
        r = imgFilterSingle(r,size, filterType,q, d)
        g = imgFilterSingle(g,size, filterType,q, d)
        b = imgFilterSingle(b,size, filterType,q,d )
        return cv.merge([r, g, b])
    else:
        raise(RuntimeError("维度错误,维度为"+str(dimension)))

def imgFilterSingle(img, size=3,filterType=const.MAX_FILTER, q=1,d=2):
    '''

    :param img:
    :param size:
    :param filter:
    :return:
    '''
    #定义filterType和method之间的映射
    typeToMethod = {
        const.MAX_FILTER : maxFilterOperator,
        const.MIN_FILTER : minFilterOperator,
        const.GEOMETRIC_MEAN_FILTER : geometricMeanOperator,
        const.HMEANS_FILTER : HMeanOperator,
        const.IHMEANS_FILTER : iHMeanOperator,
        const.MINPOINT_FILTER: minpointFilterOperator,
        const.MEDIAN_FILTER: medianFilterOperator,
        const.ALPHA_MEAN_FILTER_AMEND : alphaMeanFilterAmendOperator
    }
    method = typeToMethod.get(filterType)
    length = int(size/2)
    resultImg = np.zeros(img.shape)
    tempImg = cv.copyMakeBorder(img, length,length,length,length,cv.BORDER_DEFAULT)
    for i in range(length, tempImg.shape[0]-length):
        for j in range(length, tempImg.shape[1]-length):
            if const.IHMEANS_FILTER == filterType:
                resultImg[i - length, j - length] = method(
                    tempImg[i - length:i + length + 1, j - length:j + length + 1],q)
            elif const.ALPHA_MEAN_FILTER_AMEND == filterType:
                resultImg[i - length, j - length] = method(
                    tempImg[i - length:i + length + 1, j - length:j + length + 1],d)
            else:
                resultImg[i-length, j-length] = method(tempImg[i-length:i+length+1, j-length:j+length+1])
    return resultImg.astype(np.uint8)


path = 'C:/Users/i349006/PycharmProjects/DIP3E_CH05_Original_Images/'
fileName = 'Fig0512(b)(ckt-uniform-plus-saltpepr-prob-pt1).tif'
img = cv.imread(path+fileName, 0)

#resultImg = madianFilter(img, size=5)
resultImg = imgFilter(img,size=5, filterType=const.ALPHA_MEAN_FILTER_AMEND, d=5)
#resultImg = imgFilter(img,3, const.IHMEANS_FILTER,q=-1.5)
#resultImg = madianFilter(resultImg,3)

cv.namedWindow('test')
#cv.imshow('test', resultImg)
cv.imshow('test', np.hstack([img,resultImg]))
cv.waitKey(0)

cv.destroyAllWindows()

