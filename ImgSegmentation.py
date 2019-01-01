#图像分割
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage.io as io
import sys


laplaceKernel = np.array([[1,1,1],
                          [1,-8,1],
                          [1,1,1]])

prewittXKernel = np.array([[-1,-1,-1],
                           [0,0,0],
                           [1,1,1]])
prewittYKernel = np.array([[-1,0,1],
                           [-1,0,1],
                           [-1,0,1]])

sobelXKernel = np.array([[-1,-2,-1],
                           [0,0,0],
                           [1,2,1]])
sobelYKernel = np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]])

prewittXDiaKernel = np.array([[0,1,1],
                           [-1,0,1],
                           [-1,-1,0]])
prewittYDiaKernel = np.array([[-1,-1,0],
                           [-1,0,1],
                           [0,1,1]])
sobelXDiagKernel = np.array([[0,1,2],
                           [-1,0,1],
                           [-2,-1,0]])
sobelYDiagKernel = np.array([[-2,-1,0],
                           [-1,0,1],
                           [0,1,2]])

filePath = 'E:\\dataImag\\DIP3E_Original_Images_CH10\\'
def filterProcess(img, kernel):
    #自定义卷积操作
    return cv.filter2D(img, -1, kernel)

def testPointDetection():
    name = 'Fig1004(b)(turbine_blade_black_dot).tif'
    img = cv.imread(filePath + name,0)

    result = filterProcess(img, laplaceKernel)
#    result = cv.Laplacian(img, -1,ksize=3,scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    cv.namedWindow('check point')
    cv.imshow('check point', np.hstack([img, result]))
    cv.waitKey(0)
    cv.destroyAllWindows()

def testLineDetection(name,kernel = laplaceKernel):
    img = cv.imread(filePath + name, 0)

    result = filterProcess(img, kernel)
    #    result = cv.Laplacian(img, -1,ksize=3,scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')

    plt.show()

def testPrewittFilter(name):
    img = cv.imread(filePath+name,0)

    (height, width) = img.shape
    print(img.shape)
    preX = filterProcess(img, prewittXKernel)
    preY = filterProcess(img, prewittYKernel)

    preImg = preX + preY
    plt.figure(figsize=(width/100*4, height/100*4))
    plt.subplot(1,4,1)
    plt.title('origin')
    plt.imshow(img, cmap='gray')

    plt.subplot(1,4,2)
    plt.title('gradX')
    plt.imshow(preX, cmap='gray')

    plt.subplot(1,4,3)
    plt.title('gradY')
    plt.imshow(preY, cmap='gray')

    plt.subplot(1,4,4)
    plt.title('gx+gy')
    plt.imshow(preImg, cmap='gray')

    plt.show()

    # cv.namedWindow('test', 0)
    # cv.imshow('test', np.hstack([img, preX, preY, preImg]))
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def testGradFilter(name, kernelX=sobelXKernel, kernelY=sobelYKernel):
    img = cv.imread(filePath+name,0)

    (height, width) = img.shape
    preX = filterProcess(img, kernelX)
    preY = filterProcess(img, kernelY)

    preImg = preX + preY
    plt.figure(figsize=(width/100*4, height/100*4))
    plt.subplot(1,4,1)
    plt.title('origin')
    plt.imshow(img, cmap='gray')

    plt.subplot(1,4,2)
    plt.title('gradX')
    plt.imshow(preX, cmap='gray')

    plt.subplot(1,4,3)
    plt.title('gradY')
    plt.imshow(preY, cmap='gray')

    plt.subplot(1,4,4)
    plt.title('gx+gy')
    plt.imshow(preImg, cmap='gray')

    plt.show()


def gradDetection(img, kernelX=sobelXKernel, kernelY=sobelYKernel):
    (height, width) = img.shape
    preX = filterProcess(img, kernelX)
    preY = filterProcess(img, kernelY)

    preImg = preX + preY
    plt.figure(figsize=(width/100*4, height/100*4))
    plt.subplot(1,4,1)
    plt.title('origin')
    plt.imshow(img, cmap='gray')

    plt.subplot(1,4,2)
    plt.title('gradX')
    plt.imshow(preX, cmap='gray')

    plt.subplot(1,4,3)
    plt.title('gradY')
    plt.imshow(preY, cmap='gray')

    plt.subplot(1,4,4)
    plt.title('gx+gy')
    plt.imshow(preImg, cmap='gray')

    plt.show()


def edgesMarrHildreth(img, sigma):
    """
        finds the edges using MarrHildreth edge detection method...
        :param im : input image
        :param sigma : sigma is the std-deviation and refers to the spread of gaussian
        :return:
        a binary edge image...
    """
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    print('size='+str(size))

    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1), np.arange(-size / 2 + 1, size / 2 + 1))

    normal = 1 / (2.0 * np.pi * sigma ** 2)

    kernel = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normal  # LoG filter

    kern_size = kernel.shape[0]
    log = np.zeros_like(img, dtype=float)

    # applying filter
    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i:i + kern_size, j:j + kern_size] * kernel
            log[i, j] = np.sum(window)

    log = log.astype(np.int64, copy=False)

    zero_crossing = np.zeros_like(log)

    # computing zero crossing
    for i in range(log.shape[0] - (kern_size - 1)):
        for j in range(log.shape[1] - (kern_size - 1)):
            if log[i][j] == 0:
                if (log[i][j - 1] < 0 and log[i][j + 1] > 0) or (log[i][j - 1] < 0 and log[i][j + 1] < 0) or (
                        log[i - 1][j] < 0 and log[i + 1][j] > 0) or (log[i - 1][j] > 0 and log[i + 1][j] < 0):
                    zero_crossing[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j - 1] > 0) or (log[i][j + 1] > 0) or (log[i - 1][j] > 0) or (log[i + 1][j] > 0):
                    zero_crossing[i][j] = 255

                # plotting images
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(log, cmap='gray')
    a.set_title('Laplacian of Gaussian')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(zero_crossing, cmap='gray')
    string = 'Zero Crossing sigma = '
    string += (str(sigma))
    a.set_title(string)
    plt.show()

    return zero_crossing
def testGauss():
    lenaImg = cv.imread(filePath+'lena_std.tif',0)
    gaussLena = cv.GaussianBlur(lenaImg, (5,5),0)
    plt.title("testGauss")
    plt.imshow(gaussLena, cmap='gray')
    plt.show()


def testGradSobel(img):
    Gx = cv.Sobel(img,-1,1,0,ksize=3)
    Gy = cv.Sobel(img, -1,0,1,ksize=3)
    G = np.sqrt(np.square(Gx.astype(np.float64)) + np.square(Gy.astype(np.float64)))
    cita = np.arctan2(Gy.astype(np.float64), Gx.astype(np.float64))

    fig = plt.figure()
    a = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(Gx, cmap='gray')
    a.set_title('Sobel X')
    a = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(Gy, cmap='gray')
    a.set_title('Sobel Y')
    a = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(cita, cmap='gray')
    a.set_title('cita')
    plt.show()

def NonmaximumSuppression(image,cita, gValue):
    keep = np.zeros(cita.shape)
    cita = np.abs(cv.copyMakeBorder(cita,1,1,1,1,cv.BORDER_DEFAULT))
    for i in range(1,cita.shape[0]-1):
        for j in range(1,cita.shape[1]-1):
            if cita[i][j]>cita[i-1][j] and cita[i][j]>cita[i+1][j]:
                keep[i-1][j-1] = 1
            elif cita[i][j]>cita[i][j+1] and cita[i][j]>cita[i][j-1]:
                keep[i-1][j-1] = 1
            elif cita[i][j]>cita[i+1][j+1] and cita[i][j]>cita[i-1][j-1]:
                keep[i-1][j-1] = 1
            elif cita[i][j]>cita[i-1][j+1] and cita[i][j]>cita[i+1][j-1]:
                keep[i-1][j-1] = 1
            else:
                keep[i-1][j-1] = 0
    return keep*image

houseImg = cv.imread(filePath+'lena_std.tif',0)
testGradSobel(houseImg)
# cv.namedWindow('test')
# cv.imshow('test', houseImg)
# # cv.imshow('test', np.hstack([houseImg, result]))
# cv.waitKey(0)
# cv.destroyAllWindows()