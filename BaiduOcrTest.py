from aip import AipOcr
import cv2 as cv
import base64
import os
import numpy as np

path = 'E:/dataset/'
cellPath = path+'thresCell3/'
name='rectifyTable.jpg'
cellPath2 = path+'thresCell/'
imgName = '174.jpg'

APP_ID = '15582892'
API_KEY = 'Xt6Sq24STPHp4zduLPHLSoKz'
SECRET_KEY = 'eYatFz0cUjc6BL8zGEEUe8EQI81Q5hQi'

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()




client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
# img = get_file_content(cellPath+imgName)
# img = get_file_content('D:/Image/rectifyTable.jpg')
# result = client.tableRecognitionAsync(img)
# print(result)
# requestIds = result.get('result')
# for item in requestIds:
#     requestId = item['request_id']
#     print(requestId)
#     client.getTableRecognitionResult(requestId)
#     options = {}
#     options["result_type"] = "excel"
#     res = client.getTableRecognitionResult(requestId, options)
#     print(res)
#
client.getTableRecognitionResult('15582892_909314')
options = {}
options["result_type"] = "excel"
res = client.getTableRecognitionResult('15582892_909314', options)
print(res)

# res = client.basicGeneral(img)
# for item in res.get('words_result'):
#     print(item['words'])
# fileList = os.listdir(cellPath)
# fileList.sort()
# print(len(fileList))
# for fileName in fileList:
#     print(fileName)
# for fileName in fileList:
#     img = get_file_content(cellPath + fileName)
#     res = client.basicGeneral(img)
#     for item in res.get('words_result'):
#         print(item['words'])
# sourceTable = get_file_content(path+name)
# sourceTable = cv.imread(path+name)
# print(type(sourceTable))
# img = cv.imencode('.jpg', sourceTable)[1]
# imageCode = str(base64.b64encode(img))[2:-1]
# imageCode = base64.b64encode(img)
# imageCode = bytes(imageCode)
# print(type(img))
# print(type(base64.b64encode(img)))
# print(type(imageCode))
# res = client.basicGeneral(imageCode)
# print(type(res))



