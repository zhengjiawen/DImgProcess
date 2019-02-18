#检测table的轮廓

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = 'E:/dataset/'
name='testTable.jpg'

sourceTable = cv.imread(path+name)
grayTable = cv.cvtColor(sourceTable, cv.COLOR_BGR2GRAY)
thresTable = cv.adaptiveThreshold(grayTable, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -2)

img_median = cv.medianBlur(grayTable, 3)
thresMedianImg = cv.adaptiveThreshold(img_median, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

edge = cv.Canny(thresMedianImg, 50,200)
plt.figure(figsize=(36, 36))

plt.subplot(121)
plt.imshow(edge,'gray')

lines = cv.HoughLinesP(edge, 1, np.pi/180, 30, minLineLength=60,maxLineGap=10)
lines1 = lines[:,0,:]#提取为二维
for x1,y1,x2,y2 in lines1[:]:
    cv.line(sourceTable,(x1,y1),(x2,y2),(255,0,0),1)
plt.subplot(122)
plt.imshow(sourceTable,)

plt.show()
# contours, hierarchy = cv.findContours(thresMedianImg,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(grayTable, contours,-1,(0,0,255),3)


cv.namedWindow('table')
cv.imshow('table', edge)

cv.waitKey(0)
cv.destroyAllWindows()