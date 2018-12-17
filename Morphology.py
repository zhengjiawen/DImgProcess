#形态学操作
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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

def openOperation(img):
    '''
    开操作
    :param img:
    :return:
    '''
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
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



testOpenAndCloseOperation()
