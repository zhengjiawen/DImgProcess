from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import operator

from sklearn.decomposition import PCA
from sklearn.neighbors  import KNeighborsClassifier

data = fetch_olivetti_faces()
allImgs = data.data
target = data.target

trainNum = 7
testNum = 2
crossValidateNum = 1
dataNum = allImgs.shape[0]
dataGroup = int(dataNum/10)
oriDim = allImgs.shape[1]
N = 100

train_data = np.empty((dataGroup*trainNum, oriDim))
train_label = np.empty(dataGroup*trainNum)

test_cross_validation_data = np.empty((dataGroup*crossValidateNum, oriDim))
test_cross_validation_label = np.empty((dataGroup*crossValidateNum))

test_data = np.empty((dataGroup*testNum, oriDim))
test_label = np.empty(dataGroup*testNum)

def prepareDataSet():
    for i in range(40):
        train_data[i * trainNum:i * trainNum + trainNum] = allImgs[i * 10:i * 10 + trainNum]
        train_label[i * trainNum:i * trainNum + trainNum] = target[i * 10:i * 10 + trainNum]

        test_cross_validation_data[i*crossValidateNum:i*crossValidateNum+crossValidateNum] = allImgs[i*10+trainNum+testNum : i*10+trainNum+testNum+crossValidateNum]
        test_cross_validation_label[i*crossValidateNum:i*crossValidateNum+crossValidateNum] = target[i*10+trainNum+testNum : i*10+trainNum+testNum+crossValidateNum]

        test_data[i * testNum: i * testNum + testNum] = allImgs[i * 10 + trainNum : i*10+trainNum+testNum]
        test_label[i * testNum: i * testNum + testNum] = target[i * 10 + trainNum : i*10+trainNum+testNum]


def compute_eigenValues_eigenVectors(arr):
    # 计算协方差矩阵的特征值和特征向量，按从大到小顺序排列

    avgArr = np.mean(arr,1)
    avgArr = avgArr.reshape(avgArr.shape[0],1)
    normArr = arr - avgArr
    #计算arr'T * arr

    # covMat = np.cov(normArr, rowvar=1)
    temp=np.dot(normArr.T, normArr)
    eigenValues,eigenVectors=np.linalg.eig(temp)
#    print(eigenVectors.shape)
    #将数值从大到小排序
    idx=np.argsort(-eigenValues)
    eigenValues=eigenValues[idx]
    #特征向量按列排
    eigenVectors=eigenVectors[:,idx]

    return eigenValues,np.dot(normArr,eigenVectors),eigenVectors

def knnOnce(eigenMat,eigenTrainMat,eigenTrainLabel,k):


    dis = np.sum(np.power(eigenMat.reshape(eigenMat.shape[0],1) - eigenTrainMat,2), axis=0)
    sortedDistIndicies = dis.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = eigenTrainLabel[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def knnModel(trainData, trainLabel, testData, testLabel, k):
    prepareDataSet()
    num = testData.shape[0]
    avgTrainData = np.mean(trainData, 0)
    eigenTrainValues, eigenTrainMat, eigenTrainVec = compute_eigenValues_eigenVectors(trainData.T)
    print('eigenMat:'+str(eigenTrainMat.shape))
    print(eigenTrainVec.shape)

    eigenTrainMat = eigenTrainMat[:, 0:N]
    trainVec = np.dot(eigenTrainMat.T, (trainData - avgTrainData).T)
    print(eigenTrainVec.shape)

#    eigenValues, eigenMats, eigenVectors, eigenLabel = compute_eigenValues_eigenVectors(testData.T, testLabel)
#    testVec = np.dot(eigenTrainMat.T, (testData-avgTrainData).T)
    testVec = np.dot(eigenTrainMat.T, (testData - avgTrainData).T)
    print(testVec.shape)
    errorCount = 0

    for i in range(num):
        result = knnOnce(testVec[:, i],trainVec,trainLabel,k)
        print('predicct: '+str(result))
        print('true: '+str(testLabel[i]))
        if result != testLabel[i]:
            errorCount+=1

    print("accuracy: "+str(float(1.0- errorCount/num)))

def LDA(dataset, dataLabel, targetDim):
    [n, d] = dataset.shape
    label = np.unique(dataLabel)

    meanTotal = np.mean(dataset, axis=0)
    Sw = np.zeros((d,d), dtype=np.float32)
    Sb = np.zeros((d,d), dtype=np.float32)

    X_classify = {}
    for l in label:
        X1 = np.array([dataset[i] for i in range(len(dataset)) if dataLabel[i] == l])
        X_classify[l] = X1

    for i in label:

        Xi = X_classify[i]
        meanClassify = np.mean(Xi, axis=0)
        Sw = Sw + np.dot((Xi - meanClassify).T, (Xi - meanClassify))
        Sb = Sb + n*np.dot((meanClassify - meanTotal).T, (meanClassify-meanTotal))
    Sw1 = (Sw+Sw.T)/2
    Sw1 = np.linalg.inv(Sw1)
    eigenValue, eigenVector = np.linalg.eig(Sw1*Sb)
    idx = np.argsort(- eigenValue.real)
    eigenValue, eigenVector = eigenValue[idx], eigenVector[idx]
    eigenValue = np.array(eigenValue[0:targetDim].real, dtype=np.float32, copy=True)
    eigenVector = np.array(eigenVector[0:, 0:targetDim].real, dtype=np.float32, copy=True)
    return  eigenValue, eigenVector

def fisherface(dataset, label, target):
#    y = np.asarray(label)
    num, dim = dataset.shape
    c = len(np.unique(target))
    meanData = np.mean(dataset, axis=0)
    eigenValuePca, eigenVectorPca, eigenVectorPca_ori = compute_eigenValues_eigenVectors(dataset.T)
    tempEigenVector = np.dot(dataset - meanData, eigenVectorPca)
    eigenValueLda, eigenVectorLda, = LDA(tempEigenVector, label, target)

    eigenVectors = np.dot(eigenVectorPca, eigenVectorLda)
    return eigenValueLda, eigenVectors, meanData


def knnModelWithLDA(trainData, trainLabel, testData, testLabel, k, n):
    prepareDataSet()
    num = testData.shape[0]
    avgTrainData = np.mean(trainData, 0)
    # LDA
    eigenValues, eigenVector, eigenVectorPca_ori = fisherface(train_data, train_label, n)
    trainLda = np.dot(trainData, eigenVector)
    testLda = np.dot(testData, eigenVector)
    errorCount = 0

    for i in range(num):
        result = knnOnce(testLda[i,:],trainLda.T,trainLabel,k)
        # print('predicct: '+str(result))
        # print('true: '+str(testLabel[i]))
        if result != testLabel[i]:
            errorCount+=1
    accuracy = float(1.0- errorCount/num)
    print("accuracy: "+str(accuracy))
    return accuracy

def plotAccuracy():
    accuracyList = []
    for i in range(1,300):
        accuracy = knnModelWithLDA(train_data, train_label, test_cross_validation_data, test_cross_validation_label,k=3,n=i)
        accuracyList.append(accuracy)

    plt.plot(accuracyList)
    plt.show()


plotAccuracy()
# knnModel(train_data, train_label, train_data, train_label,5)


def testInLib():
    prepareDataSet()
    #先训练PCA模型
    PCA=PCA(n_components=100).fit(train_data)
    #返回测试集和训练集降维后的数据集
    x_train_pca = PCA.transform(train_data)
    x_test_pca = PCA.transform(test_data)

    knn=KNeighborsClassifier(n_neighbors=6)
    knn.fit(x_train_pca, train_label)

    y_test_predict=knn.predict(x_test_pca)
    #输出
    count = 0
    for i in range(len(y_test_predict)):
        if y_test_predict[i] != test_label[i]:
            count +=1
    print(float(count/test_label.shape[0]))