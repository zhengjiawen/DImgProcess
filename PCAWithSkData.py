from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

data = fetch_olivetti_faces()
allImgs = data.data
target = data.target

trainNum = 7
testNum = 2
crossValidateNum = 1
dataNum = allImgs.shape[0]
dataGroup = int(dataNum/10)
oriDim = allImgs.shape[1]
N = 280
C = 40

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
        test_cross_validation_data[i*crossValidateNum:i*crossValidateNum+crossValidateNum] = target[i*10+trainNum+testNum : i*10+trainNum+testNum+crossValidateNum]

        test_data[i * testNum: i * testNum + testNum] = allImgs[i * 10 + trainNum : i*10+trainNum+testNum]
        test_label[i * testNum: i * testNum + testNum] = target[i * 10 + trainNum : i*10+trainNum+testNum]


def compute_eigenValues_eigenVectors(arr):
    # 计算协方差矩阵的特征值和特征向量，按从大到小顺序排列

    avgArr = np.mean(arr,1)
    avgArr = avgArr.reshape(avgArr.shape[0],1)
    normArr = arr - avgArr
    #计算arr'T * arr

    temp=np.dot(normArr.T, normArr)
    eigenValues,eigenVectors=np.linalg.eig(temp)
#    print(eigenVectors.shape)
    #将数值从大到小排序
    idx=np.argsort(-eigenValues)
    eigenValues=eigenValues[idx]
    #特征向量按列排
    eigenVectors=eigenVectors[:,idx]

    return eigenValues,np.dot(normArr,eigenVectors),eigenVectors

def reconstruct(eigenVectors, eigenMat, meanImg, n=10):
    img = np.dot(eigenVectors[:,  0:n], eigenMat[0:n].reshape(n,1))+meanImg.reshape(meanImg.shape[0],1)
    img = img.reshape(64,64)
    return img

def testMeanFace():
    avgImg = np.mean(train_data, 0)

    avgImg = avgImg.reshape(64,64)
    plt.imshow(avgImg, cmap='gray')
    plt.show()

def testFirstFiveEigenFace():
    eigenValues, eigenMat, eigenVectors = compute_eigenValues_eigenVectors(train_data.T)
    img1 = eigenMat[:, 0].reshape(64, 64)
    plt.subplot(1, 5, 1)
    plt.imshow(img1, cmap='gray')

    img2 = eigenMat[:, 1].reshape(64, 64)
    plt.subplot(1, 5, 2)
    plt.imshow(img2, cmap='gray')

    img3 = eigenMat[:, 2].reshape(64, 64)
    plt.subplot(1, 5, 3)
    plt.imshow(img3, cmap='gray')

    img4 = eigenMat[:, 3].reshape(64, 64)
    plt.subplot(1, 5, 4)
    plt.imshow(img4, cmap='gray')

    img5 = eigenMat[:, 4].reshape(64, 64)
    plt.subplot(1, 5, 5)
    plt.imshow(img5, cmap='gray')

    plt.show()

def compute_img_mes(predictImgs, srcImgs):
    '''
    计算均方误差
    :param predictImgs: 行是维度，列是样本数
    :param srcImgs:
    :return:
    '''

    dim, nums = predictImgs.shape[:2]

    return np.sum(np.power(predictImgs-srcImgs, 2))/nums

def testReconstructImg():
    eigenValues, eigenMats, eigenVectors, eigenLabel = compute_eigenValues_eigenVectors(train_data.T)
    avgImg = np.mean(train_data, 0)
    print(eigenMats.shape)
    print(eigenVectors.shape)
    img = reconstruct(eigenMats, eigenVectors[61], avgImg, 150)
    print(compute_img_mes(img, train_data[61].reshape(64,64)))
    plt.subplot(1,2,1)
    plt.imshow(train_data[61].reshape(64,64),cmap='gray')
    plt.subplot(1,2,2)

    plt.imshow(img, cmap='gray')

    plt.show()

def testTSNE():
    # # covertImg()
    prepareDataSet()
    avgTrainData = np.mean(train_data, 0)
    eigenTrainValues, eigenTrainMat, eigenTrainVec = compute_eigenValues_eigenVectors(train_data.T)
    eigenTrainMat = eigenTrainMat[:, 0:100]
    testVec = np.dot(eigenTrainMat.T, (test_data - avgTrainData).T)
    print(testVec.shape)

    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(testVec.T)
    # 使用PCA 进行降维处理
    # 设置画布的大小
    plt.figure(figsize=(12, 6))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=test_label)
    # plt.subplot(122)
    # plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
    plt.colorbar()
    plt.show()


def LDA_dimensionality(dataset, dataLabel, targetDim):
    '''
    X为数据集，y为label，k为目标维数
    '''
    complexNum = 0
    label_ = list(set(dataLabel))

    X_classify = {}

    for label in label_:
        X1 = np.array([dataset[i] for i in range(len(dataset)) if dataLabel[i] == label])
        X_classify[label] = X1

    meanu = np.mean(dataset, axis=0)
    meanu_classify = {}

    for label in label_:
        meanu1 = np.mean(X_classify[label], axis=0)
        meanu_classify[label] = meanu1

    #St = np.dot((X - meanu).T, X - meanu)

    Sw = np.zeros((len(meanu), len(meanu)))  # 计算类内散度矩阵
    for i in label_:
        Sw += np.dot((X_classify[i] - meanu_classify[i]).T,
                     X_classify[i] - meanu_classify[i])


    # Sb=St-Sw
    Sb = np.zeros((len(meanu), len(meanu)))  # 计算类间散度矩阵
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((meanu_classify[i] - meanu).reshape(
            (len(meanu), 1)), (meanu_classify[i] - meanu).reshape((1, len(meanu))))
    temp = np.dot(np.linalg.inv(Sw),Sb)
    temp = (temp+temp.T)/2
    eig_vals, eig_vecs = np.linalg.eig(temp)  # 计算Sw-1*Sb的特征值和特征矩阵
    # print("==============")
    # print(eig_vals)

    idx = np.argsort(-eig_vals.real)
    eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:,idx]
    # 排序,取特征向量
    eig_vals = np.array(eig_vals[0:targetDim].real, dtype=np.float32, copy=True)
    topk_eig_vecs = np.array(eig_vecs[0:,0:targetDim].real, dtype=np.float32, copy=True)

    return  eig_vals,topk_eig_vecs

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



def testFirstFiveFisherFace():
    eigenValues, eigenVector, eigenVectorPca_ori = fisherface(train_data, train_label, 30)
    print(eigenVector.shape)
    img1 = eigenVector[:, 0].reshape(64, 64)
    plt.subplot(1, 5, 1)
    plt.imshow(img1, cmap='gray')

    img2 = eigenVector[:, 1].reshape(64, 64)
    plt.subplot(1, 5, 2)
    plt.imshow(img2, cmap='gray')

    img3 = eigenVector[:, 2].reshape(64, 64)
    plt.subplot(1, 5, 3)
    plt.imshow(img3, cmap='gray')

    img4 = eigenVector[:, 3].reshape(64, 64)
    plt.subplot(1, 5, 4)
    plt.imshow(img4, cmap='gray')

    img5 = eigenVector[:, 4].reshape(64, 64)
    plt.subplot(1, 5, 5)
    plt.imshow(img5, cmap='gray')

    plt.show()

def testLDATSNE():
    prepareDataSet()
    avgTrainData = np.mean(train_data, 0)
    eigenValues, eigenVector, eigenVectorPca_ori = fisherface(train_data, train_label, 30)
    testVec = np.dot(test_data, eigenVector)



    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(testVec)
    # 使用PCA 进行降维处理
    # 设置画布的大小
    plt.figure(figsize=(12, 6))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=test_label)
    # plt.subplot(122)
    # plt.scatter(pca[:, 0], pca[:, 1], c=iris.target)
    plt.colorbar()
    plt.show()

prepareDataSet()
testLDATSNE()

