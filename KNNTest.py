import numpy as np
from PIL import Image
import pickle
import operator
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

imgPath = 'E:/dataset/'
trainNum = 7
testNum = 2
crossValidateNum = 1
dataNum = 40
N = 150

train_data = np.empty((dataNum*trainNum, 2679))
train_label = np.empty(dataNum*trainNum)

test_cross_validation_data = np.empty((dataNum*crossValidateNum, 2679))
test_cross_validation_label = np.empty((dataNum*crossValidateNum))

test_data = np.empty((dataNum*testNum, 2679))
test_label = np.empty(dataNum*testNum)



def covertImg():
    imgName = 'olivettifaces.gif'
    # 读取原始图片并转化为np.ndarray，将灰度值由0～256转换到0～1
    img = Image.open(imgPath+imgName)
    img_ndarray = np.asarray(img, dtype='float64')/256

    # 图片大小时1190*942，一共20*20个人脸图，故每张人脸图大小为（1190/20）*（942/20）即57*47=2679
    # 将全部400个样本存储为一个400*2679的数组，每一行即代表一个人脸图，并且第0～9、10～19、20～29...行分别属于同个人脸
    # 另外，用olivettifaces_label表示每一个样本的类别，它是400维的向量，有0～39共40类，代表40个不同的人。
    olivettifaces = np.empty((400, 2679))
    for row in range(20):
        for column in range(20):
            olivettifaces[row * 20 + column] = np.ndarray.flatten(
                img_ndarray[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47])

    # 建一个<span style="font-family: SimSun;">olivettifaces_label</span>
    olivettifaces_label = np.empty(400)
    for label in range(40):
        olivettifaces_label[label * 10:label * 10 + 10] = label
    olivettifaces_label = olivettifaces_label.astype(np.int)

    # 保存olivettifaces以及olivettifaces_label到olivettifaces.pkl文件
    write_file = open(imgPath+'olivettifaces.pkl', 'wb')
    pickle.dump(olivettifaces, write_file, -1)
    pickle.dump(olivettifaces_label, write_file, -1)
    write_file.close()

def prepareDataSet():
    read_file=open(imgPath+'olivettifaces.pkl','rb')
    faces = pickle.load(read_file)
    label = pickle.load(read_file)
    read_file.close()

    for i in range(40):
        train_data[i * trainNum:i * trainNum + trainNum] = faces[i * 10:i * 10 + trainNum]
        train_label[i * trainNum:i * trainNum + trainNum] = label[i * 10:i * 10 + trainNum]

        test_cross_validation_data[i*crossValidateNum:i*crossValidateNum+crossValidateNum] = faces[i*10+trainNum+testNum : i*10+trainNum+testNum+crossValidateNum]
        test_cross_validation_data[i*crossValidateNum:i*crossValidateNum+crossValidateNum] = label[i*10+trainNum+testNum : i*10+trainNum+testNum+crossValidateNum]

        test_data[i * testNum: i * testNum + testNum] = faces[i * 10 + trainNum : i*10+trainNum+testNum]
        test_label[i * testNum: i * testNum + testNum] = label[i * 10 + trainNum : i*10+trainNum+testNum]

def compute_eigenValues_eigenVectors(arr,label):
    # 计算协方差矩阵的特征值和特征向量，按从大到小顺序排列

    avgArr = np.mean(arr,1)
    avgArr = avgArr.reshape(avgArr.shape[0],1)
    normArr = arr - avgArr
    #计算arr'T * arr

    temp=np.dot(normArr.T, normArr)
    eigenValues,eigenVectors=np.linalg.eig(temp)
    #将数值从大到小排序
    idx=np.argsort(-eigenValues)
    eigenValues=eigenValues[idx]
    #特征向量按列排
    eigenVectors=eigenVectors[:,idx]
    #label排序
    eigenVectorsLabel = label.copy()
    eigenVectorsLabel = eigenVectorsLabel[idx]
    return eigenValues,np.dot(normArr,eigenVectors),eigenVectors, eigenVectorsLabel



def knnOnce(eigenMat,eigenTrainMat,eigenTrainLabel,k):


    dis = np.sum(np.power(eigenMat.reshape(eigenMat.shape[0],1) - eigenTrainMat,2), axis=0)
    sortedDistIndicies = dis.argsort()
    print(dis[sortedDistIndicies[0]])

    classCount = {}

    for i in range(k):
        voteIlabel = eigenTrainLabel[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def knnModel(trainData, trainLabel, testData, testLabel):
    prepareDataSet()
    num = testData.shape[0]

    eigenTrainValues, eigenTrainMat, eigenTrainVec, eigenTrainLabel = compute_eigenValues_eigenVectors(trainData.T, trainLabel)
    eigenTrainMat = eigenTrainMat[0:N,:]

    eigenValues, eigenMats, eigenVectors, eigenLabel = compute_eigenValues_eigenVectors(testData.T, testLabel)
    errorCount = 0

    for i in range(num):
        result = knnOnce(eigenMats[0:N, i],eigenTrainMat,eigenTrainLabel,5)
        if result != eigenLabel[i]:
            errorCount+=1

    print(str(float(errorCount/num)))


knnModel(train_data, train_label, test_data, test_label)
