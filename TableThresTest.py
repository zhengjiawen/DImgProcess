import cv2 as cv
import numpy as np

path = 'E:/dataset/'
savePath = "thresImg/"
name='testTable.jpg'

def dilaAndErode(img, kernel):
    img = cv.dilate(img, kernel)
    img  = cv.erode(img, kernel)
    return img


sourceTable = cv.imread(path+name)
grayTable = cv.cvtColor(sourceTable, cv.COLOR_BGR2GRAY)

img_median = cv.medianBlur(grayTable, 3)
ret, thresImg = cv.threshold(img_median, 0, 255, cv.THRESH_OTSU)




scale = 10
horizontalSize = thresImg.shape[1] // scale
verticalSize = thresImg.shape[0] // scale
horKernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize,1))
vecKernel = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
# print(horKernel)

horImg = dilaAndErode(thresImg, horKernel)
vecImg = dilaAndErode(thresImg, vecKernel)


# edges = cv.Canny(thresImg, 50, 200)
lines = cv.HoughLines(grayTable, 1, np.pi/180, 2000)
lines1 = lines[:,0,:]#提取为二维
for rho,theta in lines1[:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(sourceTable,(x1,y1),(x2,y2),(0,0,255),1)



cv.namedWindow('test')
cv.imshow('test', thresImg)
cv.waitKey(0)
cv.destroyAllWindows()

# for i in range(1,255):
#     thresImg = cv.threshold(img_median, i, 255, cv.THRESH_OTSU)
# for i in range(3,100,2):
#     thresMedianImg = cv.adaptiveThreshold(img_median, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, i, -2)
#     for j in range(1,30):
#         scale = j
#         horizontalSize = thresMedianImg.shape[1] // scale
#         verticalSize = thresMedianImg.shape[0] // scale
#         horKernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
#         vecKernel = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))
#
#         horImg = dilaAndErode(thresMedianImg, horKernel)
#         vecImg = dilaAndErode(thresMedianImg, vecKernel)
#         cv.imwrite(path+'horImg/'+str(i)+'_'+str(j)+'.jpg', horImg)
#         cv.imwrite(path+'vecImg/'+str(i)+'_'+str(j)+'.jpg', vecImg)