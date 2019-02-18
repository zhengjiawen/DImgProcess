import cv2 as cv
import numpy as np

path = 'E:/dataset/'
name='testClose.jpg'
dotImgName = 'dotImg.jpg'
tableName='rectifyTable.jpg'

testFlag = False
debug = False

img = cv.imread(path+dotImgName, 0)
table = cv.imread(path+tableName)
# print(img.shape)
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 180))
# kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (1, 180))
# print(kernel.shape)
# binary = cv.erode(img, kernel)
# binary = cv.dilate(binary, kernel2)
# cv.namedWindow('table')
# cv.imshow('table', binary)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

# 找左上角第一个白色像素
def findLeftWhitePix(img):
    # 取图片前100列，找第一个白色像素坐标
    for i in range(img.shape[0]):
        for j in range(100):
            if img[i,j] == 255:
                return (i, j)

    return (-1, -1)

# 找当前像素右边第一个白色像素
def findRightWhitePix(img):
    # 取图片前100列，找第一个白色像素坐标
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == 255:
                return (i, j)

    return (-1, -1)

def drawDotPic(shape, pos, width, height):
    img = np.zeros(shape)
    for i in range(pos[0], img.shape[0], height):
        for j in range(pos[1], img.shape[1], width):
            img[i,j] = 255

    return img

def drawOriginDot(img, shape, pos, width, height):
    for i in range(pos[0], img.shape[0], height):
        for j in range(pos[1], img.shape[1], width):
            # print(img[i,j])
            img[i,j] = [0,0,255]

    return img

def writeSplitImg(img, shape, pos, width, height):
    rowList = []
    count = 0
    for i in range(pos[0], img.shape[0], height):
        cellList = []
        for j in range(pos[1], img.shape[1], width):

            if i+height > img.shape[0] or j + width > img.shape[1]:
                break;
            count = count +1
            cell = img[i+2:i+height-8, j+10:j+width-10]
            ret, thresImg = cv.threshold(cell, 0, 255, cv.THRESH_OTSU)
            cv.imwrite(path + 'thresCell3/' + str(count)+ '.jpg', thresImg)
            cellList.append(cell)

        if len(cellList) != 0:
            rowList.append(cellList)


    return rowList


def recogCell(rowList):
    count = 0
    for i in range(len(rowList)):
        cellList = rowList[i]
        for j in range(len(cellList)):
            count = count + 1

    return count


if testFlag == False:
    grayTable = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
    img_median = cv.medianBlur(grayTable, 3)
    # thresMedianImg = cv.adaptiveThreshold(img_median, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 53, -2)
    # ret, thresMedianImg = cv.threshold(img_median, 0, 255, cv.THRESH_OTSU)
    pos = findLeftWhitePix(img)
    #第一个点下面竖直的图
    img2 = img[pos[0]+10:,:]
    pos2 = findLeftWhitePix(img2)
    height = pos2[0]+10
    #第一个点右边横着的图
    img3 = img[pos[0]:, pos[1]+100:]
    pos3 = findRightWhitePix(img3)
    width = pos3[1]+100

    dotImg = drawDotPic(img.shape, pos, width, height)
    dotImg = dotImg.astype(np.uint8)

    image = drawOriginDot(table, img.shape, pos, width, height)
    rowList = writeSplitImg(img_median, img.shape, pos, width, height)
    print(len(rowList))
    print(len(rowList[0]))
    print(recogCell(rowList))
    # cv.namedWindow('table')
    #     # cv.imshow('table', dotImg)
    #     #
    #     # cv.waitKey(0)
    #     # cv.destroyAllWindows()


    if debug == True:
        print(pos)
        print(pos2)
        print(pos3)
        print('height: '+str(height))
        print('width:' +str(width))
        print(dotImg)

if testFlag == True:
    a = '一般工商业'
    b = '般工商业'

    if b.startswith('般'):
        print(b)
    if a.startswith('般'):
        print(a)
