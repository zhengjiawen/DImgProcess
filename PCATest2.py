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

def pca(X, y, target):
    [n, d] = X.shape
    mu = X.mean(axis=0)
    normX = X - mu
    temp = np.dot(normX, normX.T)
    [eigenValues, eigenVectors] = np.linalg.eigh(temp)
    eigenVectors = np.dot(normX.T, eigenVectors)
    for  i in range(n):
        eigenVectors[:, i] = eigenVectors[:,i]/np.linalg.norm(eigenVectors[:,i])

    idx = np.argsort(-eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    print("eigenVector shape: "+str(eigenVectors.shape))
    eigenValues = eigenValues[0:target].copy()
    eigenVectors = eigenVectors[:, 0:target].copy()

    return eigenValues, eigenVectors, mu

def testFirstFiveEigenFace():
    eigenValues, eigenVector, mu = pca(train_data, train_label, 100)
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

def LDA(dataset, dataLabel, targetDim):
#    dataLabel = np.asarray(dataLabel)
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
#    Sw1 = (Sw+Sw.T)/2
#    Sw1 = np.linalg.inv(Sw1)
    eigenValue, eigenVector = np.linalg.eig(Sw*Sb)
    idx = np.argsort(- eigenValue.real)
    eigenValue, eigenVector = eigenValue[idx], eigenVector[idx]
    eigenValue = np.array(eigenValue[0:targetDim].real, dtype=np.float32, copy=True)
    eigenVector = np.array(eigenVector[0:, 0:targetDim].real, dtype=np.float32, copy=True)
    return  eigenValue, eigenVector


def fisherface(dataset, label, target):
#    y = np.asarray(label)
    num, dim = dataset.shape
    c = len(np.unique(target))

    eigenValuePca, eigenVectorPca, muPca = pca(dataset, label, num-c)
    tempEigenVector = np.dot(dataset - muPca, eigenVectorPca)
    eigenValueLda, eigenVectorLda, = LDA(tempEigenVector, label, target)

    eigenVectors = np.dot(eigenVectorPca, eigenVectorLda)
    return eigenValueLda, eigenVectors, muPca




def testFirstFiveFisherFace():
    eigenValues, eigenVector, mu = fisherface(train_data, train_label, 30)
    print(eigenValues.shape)
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

prepareDataSet()
# testFirstFiveEigenFace()
testFirstFiveFisherFace()