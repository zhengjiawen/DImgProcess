import cv2 as cv
import numpy as np

path = 'E:/dataset/'
name='rectifyTable.jpg'

def dilaAndErode(img, kernel):
    img = cv.dilate(img, kernel)
    img = cv.erode(img, kernel)

    return img

# 收缩点团为单像素点（3×3）
def isolate(img):
    idx=np.argwhere(img<1)
    rows,cols=img.shape

    for i in range(idx.shape[0]):
        c_row=idx[i,0]
        c_col=idx[i,1]
        if c_col+1<cols and c_row+1<rows:
            img[c_row,c_col+1]=1
            img[c_row+1,c_col]=1
            img[c_row+1,c_col+1]=1
        if c_col+2<cols and c_row+2<rows:
            img[c_row+1,c_col+2]=1
            img[c_row+2,c_col]=1
            img[c_row,c_col+2]=1
            img[c_row+2,c_col+1]=1
            img[c_row+2,c_col+2]=1
    return img

#找左上角第一个白色像素
def findLeftWhitePix(img):
    #取图片前100列，找第一个白色像素坐标
    for i in range(img.shape[0]):
        for j in range(100):
            if img[i][j] == 255:
                return (i, j)

    return (-1,-1)





sourceTable = cv.imread(path+name)
grayTable = cv.cvtColor(sourceTable, cv.COLOR_BGR2GRAY)
thresTable = cv.adaptiveThreshold(grayTable, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -2)

img_median = cv.medianBlur(grayTable, 3)
thresMedianImg = cv.adaptiveThreshold(img_median, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 53, -2)
# ret, thresMedianImg = cv.threshold(img_median, 0, 255, cv.THRESH_OTSU)

scaleHor = 20
scaleVec = 10
horizontalSize = thresMedianImg.shape[1] // scaleHor
verticalSize = thresMedianImg.shape[0] // scaleVec
horKernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize,1))
vecKernel = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
# print(horKernel)

horImg = dilaAndErode(thresMedianImg, horKernel)
vecImg = dilaAndErode(thresMedianImg, vecKernel)
# vecImg = dilaAndErode(vecImg, vecKernel)

# img = vecImg.copy()
# img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# lines = cv.HoughLines(vecImg, 1, np.pi/180, 2000)
# lines1 = lines[:,0,:]#提取为二维
# for rho,theta in lines1[:]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv.line(img,(x1,y1),(x2,y2),(255,0,0),1)

horImg = cv.bitwise_not(horImg)
vecImg = cv.bitwise_not(vecImg)
img = cv.bitwise_and(horImg, vecImg)

# colImg = np.zeros(vecImg.shape)
#
#
#
# img = isolate(img)
# img = (horImg+vecImg)/2
# img = cv.cvtColor(horImg, cv.COLOR_GRAY2BGR)
# # img = vecImg.copy()
# lines = cv.HoughLines(img,1,np.pi/180,100)
# lines1 = lines[:,0,:]#提取为二维
# print(lines1.shape)
# for rho,theta in lines1[:]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv.line(img,(x1,y1),(x2,y2),255,2)





cv.namedWindow('table')
cv.imshow('table', horImg)
# cv.imwrite(path+'test.jpg', img)
cv.waitKey(0)
cv.destroyAllWindows()