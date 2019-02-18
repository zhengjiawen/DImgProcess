import tesserocr
import pytesseract
import cv2 as cv

path = 'E:/dataset/cell/'
name='13_269.jpg'
sourceTable = cv.imread(path+name, 0)
ret, img = cv.threshold(sourceTable, 0, 255, cv.THRESH_OTSU)

text = pytesseract.image_to_string(img, lang='chi_sim')

print(text)
if text == '':
    print(type(text))


cv.namedWindow('table')
cv.imshow('table', img)
# cv.imwrite(path+'test.jpg', img)
cv.waitKey(0)
cv.destroyAllWindows()