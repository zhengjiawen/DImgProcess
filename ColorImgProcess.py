#处理彩色图像
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


path = 'C:/Users/i349006/PycharmProjects/DIP3E_Original_Images_CH06/'
rgb_scale = 255
cmyk_scale = 100

def mergeRGB(r,g,b):
    return cv.merge([b,g,r])

def testMergeRGB():
    '''
    测试把rgb分量合并，测试合并近红外分量
    :return:
    '''
    rFile = 'Fig0627(a)(WashingtonDC Band3-RED).TIF'
    gFile = 'Fig0627(b)(WashingtonDC Band2-GREEN).TIF'
    bFile = 'Fig0627(c)(1)(WashingtonDC Band1-BLUE).TIF'
    nearInfraredFile = 'Fig0627(d)(WashingtonDC Band4).TIF'

    r = cv.imread(path+rFile, 0)
    g = cv.imread(path+gFile, 0)
    b = cv.imread(path+bFile, 0)
    ni = cv.imread(path+nearInfraredFile, 0)

    colorImag = mergeRGB(r,g,b)
    niImag = mergeRGB(ni,g,b)
    cv.namedWindow('test')
    cv.imshow('test', np.hstack([colorImag,niImag]))
    cv.waitKey(0)

def testColorImagRGB():
    '''
    测试将彩色图片分为RGB并显示
    :return:
    '''
    fileName = 'Fig0630(01)(strawberries_fullcolor).tif'

    img = cv.imread(path+fileName,1)
#    zeros = np.zeros(img.shape[:2], dtype='uint8')
    b,g,r = cv.split(img)
    # red = cv.merge([zeros,zeros,r])
    # green = cv.merge([zeros,g,zeros])
    # blue = cv.merge([b,zeros,zeros])
    cv.namedWindow('test')
    cv.imshow('test', np.hstack([r,g,b]))
    cv.waitKey(0)

def RGB2CMYK(r,g,b):
    if r.all() == 0 and g.all() == 0 and b.all() == 0:
        return 0,0,0,cmyk_scale
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)
    minCMY = np.amin([c,m,y], axis=0)

    c = c - minCMY
    m = m - minCMY
    y = y - minCMY
    k = minCMY

    return (c*cmyk_scale).astype(np.uint8), (m*cmyk_scale).astype(np.uint8), (y*cmyk_scale).astype(np.uint8), (k*cmyk_scale).astype(np.uint8)

def CMYK2RGB(c,m,y,k):
    r = rgb_scale * (1.0 - (c + k) / float(cmyk_scale))
    g = rgb_scale * (1.0 - (m + k) / float(cmyk_scale))
    b = rgb_scale * (1.0 - (y + k) / float(cmyk_scale))

    return r,g,b

def testColorImgCMYK():
    '''
    测试将彩色图片分为CMYK并显示
    :return:
    '''
    fileName = 'Fig0630(01)(strawberries_fullcolor).tif'

    img = cv.imread(path + fileName, 1)
    b, g, r = cv.split(img)
    c,m,y,k = RGB2CMYK(r,g,b)
    cv.namedWindow('test')
    cv.imshow('test', np.hstack([c,m,y,k]))
    cv.waitKey(0)

def RGB2HSI(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            theta = float(np.arccos(num/den))

            if den == 0:
                    H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2*3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum

            H = H/(2*3.14159265)
            I = sum/3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H*255
            hsi_lwpImg[i, j, 1] = S*255
            hsi_lwpImg[i, j, 2] = I*255
    return hsi_lwpImg

def HSI2RGB(hsi_img):
    """
    这是将HSI图像转化为RGB图像的函数
    :param hsi_img: HSI彩色图像
    :return: RGB图像
    """
    # 保存原始图像的行列数
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    #对原始图像进行复制
    rgb_img = hsi_img.copy()
    #对图像进行通道拆分
    H,S,I = cv.split(hsi_img)
    #把通道归一化到[0,1]
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
    R,G,B = H,S,I
    for i in range(row):
        h = H[i]*2*np.pi
        #H大于等于0小于120度时
        a1 = h >=0
        a2 = h < 2*np.pi/3
        a = a1 & a2         #第一种情况的花式索引
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i]*(1+S[i]*np.cos(h)/tmp)
        g = 3*I[i]-r-b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        #H大于等于120度小于240度
        a1 = h >= 2*np.pi/3
        a2 = h < 4*np.pi/3
        a = a1 & a2         #第二种情况的花式索引
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        #H大于等于240度小于360度
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2             #第三种情况的花式索引
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1-S[i])
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255
    return rgb_img


def testColorImgHSI():
    '''
    测试将彩色图片分为HSI并显示
    :return:
    '''
    fileName = 'Fig0630(01)(strawberries_fullcolor).tif'

    img = cv.imread(path + fileName, 1)
    result = RGB2HSI(img)
    cv.namedWindow('test')
#    cv.imshow('test', result[:,:,0])
    cv.imshow('test', np.hstack([result[:,:,0],result[:,:,1],result[:,:,2]]))
    cv.waitKey(0)

def complementaryColour(img):
    '''
    计算图片的补色
    :param img:
    :return:
    '''
    dimension = len(img.shape)
    if dimension == 2:
        return complementaryColourGray(img)
    elif dimension == 3:
        b, g, r = cv.split(img)
        r = complementaryColourGray(r)
        g = complementaryColourGray(g)
        b = complementaryColourGray(b)
        return cv.merge([b,g,r])
    else:
        raise (RuntimeError("维度错误,维度为" + str(dimension)))

def complementaryColourGray(img):
    '''
    计算单通道的补色
    :param img:
    :return:
    '''
    result = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in  range(img.shape[1]):
            result[i,j] = rgb_scale - img[i,j]
    return result.astype(np.uint8)

def testComplementaryColour():
    '''
    测试补色
    :return:
    '''
    fileName = 'Fig0630(01)(strawberries_fullcolor).tif'

    img = cv.imread(path + fileName, 1)
    result = complementaryColour(img)
#    result = RGB2HSI(result)
    cv.namedWindow('test')
    cv.imshow('test', result)
#    cv.imshow('test', np.hstack([result[:,:,0],result[:,:,1],result[:,:,2]]))

    cv.waitKey(0)


def testImgEqulizeHist():
    '''
    测试彩色直方图均衡化
    :return:
    '''
    fileName = 'Fig0635(bottom_left_stream).tif'

    img = cv.imread(path + fileName, 1)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#    img_hsv = RGB2HSI(img)
    #均衡化亮度
    img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])

    #均衡化饱和度
#    img_hsv[:, :, 1] = cv.equalizeHist(img_hsv[:, :, 1])

    result = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
#    result = HSI2RGB(img_hsv)
    cv.namedWindow('test')
    cv.imshow('test', result)
    cv.waitKey(0)


def testSplitHSIImg():
    fileName = 'Fig0642(a)(jupiter_moon_original).tif'

    img = cv.imread(path + fileName, 1)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

#    print(type(s))
    plt.figure("jupiter")
    arr = s.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor = 'green', alpha = 0.75)
    plt.show()
    thresholdS = 100
#    print(thresholdS)
    retval, resultS = cv.threshold(s,thresholdS, 255.0, cv.THRESH_BINARY)
    result = h*resultS
#    resultS.astype(np.uint8)

    cv.namedWindow('test')
    cv.imshow('test', result)
    #    cv.imshow('test', np.hstack([h,s,v]))
    cv.waitKey(0)


def testNoisyColorImag():
    fileNameR = "Fig0648(a)(lenna-noise-R-gauss-mean0-var800).tif"
    fileNameG = "Fig0648(b)(lenna-noise-G-gauss-mean0-var800).tif"
    fileNameB = "Fig0648(c)(lenna-noise-B-gauss-mean0-var800).tif"

    imgR = cv.imread(path + fileNameR, 0)
    imgG = cv.imread(path + fileNameG, 0)
    imgB = cv.imread(path + fileNameB, 0)

    print("typeR:"+str(type(imgR))+"  size"+str(imgR.size))
    print("typeG:"+str(type(imgG))+"  size"+str(imgG.size))
    print("typeB:"+str(type(imgB))+"  size"+str(imgB.size))
    img = cv.merge([imgB, imgG, imgR])

    result = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.namedWindow('test')
#    cv.imshow('test', img)
    #    cv.imshow('test', np.hstack([h,s,v]))
    cv.imshow('test', np.hstack([result[:,:,0],result[:,:,1],result[:,:,2]]))

    cv.waitKey(0)

#testNoisyColorImag()


