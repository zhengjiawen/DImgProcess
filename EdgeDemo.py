import cv2 as cv
import numpy as np
import sys

debug = False
filePath = "D:/dataset/icdar2013/"
imgFolderPath = 'Challenge2_Training_Task12_Images/'
textPath = 'Challenge2_Training_Task1_GT/'
trainNum = 100
testNum = 10
# random forest model
modelPath = 'D:/model.yml.gz'
edgeBoxesMaxBox = 1000
iouThreshold = 0.5
def computeIntRect(XA1, YA1, XA2, YA2, XB1, YB1, XB2, YB2):
    #计算两个矩形相交的部分
    #求左上角和右下角坐标，返回面积
    a = max(XA1, XB1)
    b = max(YA1, YB1)
    c = min(XA2, XB2)
    d = min(YA2, YB2)

    return max(0, c-a) * max(0, d-b)

def computeIoU(rect1, rect2, intRect):
    return intRect/(rect1+rect2-intRect)

def convertDataLabel(path, labelNum=229):
    '''
    covert ICDAR2013 data label, type is text
    :param path: label path
    :param labelNum: data num, the number of icdar 2013 dataset is 229
    :return: dict, key is img name, eg. img name is 100.jpg, the key is 100
    '''
    dataPosDict = {}
    wordRegionNum = 0
    for i in range(labelNum):
        gtPath = path +'gt_'+str(i+100)+'.txt'

        with open(gtPath, 'r') as f:
            posAndValue = []
            for line in f.readlines():

                line = line.strip()
                strs = line.split( )
                posAndValue.append(strs)
                wordRegionNum = wordRegionNum + 1
                # if debug == True:
                #     print(type(strs))
                #     print(strs)
            if debug == True:

                print('posAndValue detail:')
                print('path:'+str(gtPath))
                print('len: '+str(len(posAndValue)))
        dataPosDict[str(i+100)] = posAndValue
    return dataPosDict, wordRegionNum

def edgeBoxesGroupDetection(imgFolderPath, imgNum):
    '''
    compute icdar img edgeBoxes
    :param imgFolderPath:
    :return:
    '''
    edge_detection = cv.ximgproc.createStructuredEdgeDetection(modelPath)
    edgeBoxes = {}
    for i in range(imgNum):
        imgPath = imgFolderPath + str(i + 100) + '.jpg'
        boxes = edgeBoxesDetection(edge_detection, imgPath, edgeBoxesMaxBox)
        edgeBoxes[str(i+100)] = boxes
    return edgeBoxes

def edgeBoxesDetection(edge_detection, imgPath, maxBoxes = 1000):
    '''
    compute edgeboxes in one img
    :param imgPath: img path
    :return: boxes
    '''
    im = cv.imread(imgPath)

    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(maxBoxes)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes

def computeRecall(edgeBoxes, label):
    boxWithIouDict = {}
    wordNum = 0
    for labelkey, labelValue in label.items():
        for region in labelValue:
            iouFlag = False
            XA1, YA1, XA2, YA2, word = region
            XA1, YA1, XA2, YA2 = int(XA1), int(YA1), int(XA2), int(YA2)
            boxInOneImg = []
            for box in edgeBoxes[labelkey]:
                if debug == True:
                    print('length of box:'+str(len(box)))
                    print("box type："+ str(type(box)))
                    print('box:'+ str(box))
                XB1, YB1, w, h = box
                box = box.tolist()
                innerRect = computeIntRect(XA1, YA1, XA2, YA2, XB1, YB1, XB1+w, YB1+h)
                if innerRect != 0:
                    rect1 = abs((XA2-XA1)*(YA2-YA1))
                    rect2 = w*h
                    iou = computeIoU(rect1, rect1, innerRect)

                    if iou > iouThreshold:
                        if iouFlag == False:
                            iouFlag = True
                        box.append(iou)
                        boxInOneImg.append(box)
            boxWithIouDict[word] = boxInOneImg
            if iouFlag:
                wordNum = wordNum + 1
    return boxWithIouDict, wordNum



if __name__ == '__main__':
    # print('test')
    labelPath = filePath + textPath
    imgPath = filePath + imgFolderPath
    trainLabel , wordNum = convertDataLabel(labelPath, trainNum)
    edgeBoxes = edgeBoxesGroupDetection(imgPath, trainNum)
    boxWithIouDict, wordIouNum = computeRecall(edgeBoxes, trainLabel)
    recall = wordIouNum / wordNum
    if debug == True:
        print(boxWithIouDict)
        print('wordNum:'+str(wordNum))
        print('Iou > threshold word region:' + str(wordIouNum))
        print("recall:"+str(recall))

