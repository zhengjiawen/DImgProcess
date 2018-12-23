#形态学操作
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import measure,color
import sys


path = 'C:/Users/i349006/PycharmProjects/DIP3E_Original_Images_CH09/'

def testErode():
    '''
    测试腐蚀
    :return:
    '''
    fileName = 'Fig0905(a)(wirebond-mask).jpg'
    img = cv.imread(path + fileName, 0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(40,40))

    erodeImg = cv.erode(img,kernel)

    cv.namedWindow("test")
    cv.imshow("test", erodeImg)

    cv.waitKey(0)
    cv.destroyAllWindows()

def testDilate():
    '''
    测试膨胀
    :return:
    '''
    fileName = 'Fig0907(a)(text_gaps_1_and_2_pixels).jpg'
    img = cv.imread(path + fileName, 0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    print(kernel)

    erodeImg = cv.dilate(img,kernel)

    cv.namedWindow("test")
    cv.imshow("test", erodeImg)

    cv.waitKey(0)
    cv.destroyAllWindows()

def openOperation(img, size = (3,3)):
    '''
    开操作
    :param img:
    :return:
    '''
    kernel = cv.getStructuringElement(cv.MORPH_RECT, size)
    erodeImag = cv.erode(img, kernel)
    openImag = cv.dilate(erodeImag, kernel)

    return openImag

def closeOperation(img):
    '''
    闭操作
    :param img:
    :return:
    '''

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    dilateImag = cv.dilate(img, kernel)
    closeImag = cv.erode(dilateImag, kernel)
    return closeImag

def testOpenAndCloseOperation():
    '''
    测试开操作和闭操作
    :return:
    '''
    fileName = 'Fig0911(a)(noisy_fingerprint).jpg'
    img = cv.imread(path+fileName, 0)


    plt.subplot(1,3,1)
    plt.title('origin img')
    plt.imshow(img, cmap='gray')

    #开操作
    openImag = openOperation(img)
    plt.subplot(1,3,2)
    plt.title('open img')
    plt.imshow(openImag, cmap='gray')

    #闭操作
    closeImag = closeOperation(openImag)
    plt.subplot(1,3,3)
    plt.title('close img')
    plt.imshow(closeImag, cmap='gray')

    plt.show()


def testSimpleEdgeDetection():
    '''
    测试简单的边界提取
    :return:
    '''
    fileName = "Fig0914(a)(licoln from penny).jpg"
    img = cv.imread(path+fileName, 0)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    erodeImg = cv.erode(img, kernel)
    resultImg = img - erodeImg

    cv.namedWindow('test')
    cv.imshow('test', resultImg)
    cv.waitKey(0)
    cv.destroyAllWindows()

#连通分量提取
def twoPass(img):
    '''
    Two-pass法提取连通分量
    Twopass还有问题！！！
    :param img:
    :return:
    '''
    label = np.full(img.shape,10000)
    m, n = img.shape
    labelNum = 1
    labelList = []

    for i in range(m):
        for j in range(n):
            if 255 == img[i][j]:
                minNum = finMinNumInNeighbors(label, i, j)
                if minNum == 10000 :
                    label[i][j] = labelNum
                    labelList.append(0)
                    labelNum = labelNum + 1
                else:
                    label[i][j] = minNum
    print("LabelList:"+str(labelList))
    return label




def finMinNumInNeighbors(img, i, j):
    '''

    :param img:
    :param i:
    :param j:
    :return:
    '''

    m, n = -1, -1
    mEnd = 1
    nEnd = 1
    if i == 0:
        m = m+1
        print("m+1="+str(m))
    elif i == img.shape[0]-1:
        mEnd = mEnd -1
        print("mEnd-1="+str(mEnd))

    if j ==0:
        n = n +1
        print("n+1="+str(n))

    elif j == img.shape[1]-1:
        nEnd = nEnd -1
        print("nEnd-1="+str(nEnd))

    resultArray = img[i+m:i+mEnd+1, j+n:j+nEnd+1]
    print(resultArray)
    return np.min(resultArray)


# a = np.array([
#     [0,0,255,0,0,255,0],
#     [255,255,255,0,255,255,255],
#     [0,0,255,0,0,255,0],
#     [0,255,255,0,255,255,0]
#
# ])
#
# print(twoPass(a))


#编写一个函数来生成原始二值图像
def microstructure(l=256):
    n = 5
    x, y = np.ogrid[0:l, 0:l]  #生成网络
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)  #随机数种子
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l/(4.*n)) #高斯滤波
    return mask > mask.mean()

def testConnectedComponentsInLib():
    data = microstructure(l=128) * 1  # 生成测试图片

    labels = measure.label(data, connectivity=2)  # 8连通区域标记
    print(labels.shape)
    dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
    print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(data, plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(dst, interpolation='nearest')
    ax2.axis('off')

    fig.tight_layout()
    plt.show()

def seedFilling(img):
    label = np.full(img.shape, 0)
    labelNum = 1

    m, n = img.shape
    print(img.shape)

    for i in range(m):
        for j in range(n):
            if 255 == img[i][j]:
                if label[i][j] == 0:
                    seedFillingOnce(img, i, j, label, labelNum)
                    labelNum = labelNum + 1

    return label


def seedFillingOnce(img, m, n, label, labelNum):
    '''
    4连通
    :param img:
    :param m:
    :param n:
    :return:
    '''
    stack = []
    stack.append((m, n))

    while len(stack) != 0:
        print(len(stack))
        (i, j) = stack.pop()
        print("op:"+str(i)+","+str(j))
        label[i][j] = labelNum
        x, xEnd, y, yEnd = findNeighbors(img, i, j)

        for l in range(x, xEnd+1):
            for r in range(y, yEnd+1):
                if img[i+l][j+r] == 255 and label[i+l][j+r] == 0:
                    stack.append((i+l, j+r))


def findNeighbors(img, i, j):

    m, n = -1, -1
    mEnd = 1
    nEnd = 1
    if i == 0:
        m = m + 1
        # print("m+1=" + str(m))
    elif i == img.shape[0] - 1:
        mEnd = mEnd - 1
        # print("mEnd-1=" + str(mEnd))

    if j == 0:
        n = n + 1
        # print("n+1=" + str(n))

    elif j == img.shape[1] - 1:
        nEnd = nEnd - 1
        # print("nEnd-1=" + str(nEnd))

    # resultArray = img[i + m:i + mEnd + 1, j + n:j + nEnd + 1]
    # print(resultArray)
    return m, mEnd, n, nEnd


path2 = 'E:/lesson/DIP3E_Original_Images_CH09/'

def testTopHat():
    '''
    顶帽操作
    :return:
    '''
    name = 'Fig0940(a)(rice_image_with_intensity_gradient).jpg'
    img = cv.imread(path2+name, 0)
    ret, thImg = cv.threshold(img, 150, 255, cv.THRESH_OTSU)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (80,80))
    erodeImag = cv.erode(img, kernel)
    openImg = cv.dilate(erodeImag, kernel)

    topHatImg = img - openImg
    ret, topHatImg = cv.threshold(topHatImg, 150, 255, cv.THRESH_OTSU)
    cv.namedWindow('topHat')
    cv.imshow('topHat', np.hstack([img, thImg,topHatImg]))
    cv.waitKey(0)
    cv.destroyAllWindows()

testTopHat()
# name = 'Fig0940(a)(rice_image_with_intensity_gradient).jpg'
# img = cv.imread(path2+name, 0)
# #ret, thImg = cv.threshold(img, 150, 255, cv.THRESH_OTSU)
#
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (80,80))
# # kernel = np.ones((5,5), np.uint8)
# dst = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
# ret, dst = cv.threshold(dst, 150, 255, cv.THRESH_OTSU)
# cv.namedWindow('topHat')
# cv.imshow('topHat', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

# a = np.array([
#     [0,0,255,0,0,255,0],
#     [255,255,255,0,255,255,255],
#     [0,0,255,0,0,255,0],
#     [0,255,255,0,255,255,0]
#
# ])
#
# print(seedFilling(a))

# b = []
# print(len(b))
# b.append((1,2))
# print(len(b))
# b.append((3,4))
# print(len(b))
# b.pop()
# print(len(b))
