# 读取人脸库olivettifaces，并存储为pkl文件
import numpy as np
from PIL import Image
import pickle
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

imgPath = 'E:/dataset/'
trainNum = 7
testNum = 3
dataNum = 40

train_data = np.empty((dataNum*trainNum, 2679))
train_label = np.empty(dataNum*trainNum)

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


def testFaceImg():
    read_file=open(imgPath+'olivettifaces.pkl','rb')
    faces=pickle.load(read_file)
    read_file.close()
    img1=faces[1].reshape(57,47)
    pylab.imshow(img1)
    pylab.gray()
    pylab.show()


def prepareDataSet():
    read_file=open(imgPath+'olivettifaces.pkl','rb')
    faces = pickle.load(read_file)
    label = pickle.load(read_file)
    read_file.close()

    for i in range(40):
        train_data[i * trainNum:i * trainNum + trainNum] = faces[i * 10:i * 10 + trainNum]
        train_label[i * trainNum:i * trainNum + trainNum] = label[i * 10:i * 10 + trainNum]

        test_data[i * testNum: i * testNum + testNum] = faces[i * 10 + trainNum : i*10+trainNum+testNum]
        test_label[i * testNum: i * testNum + testNum] = label[i * 10 + trainNum : i*10+trainNum+testNum]


def compute_mean_array(arr):
    #计算均值数组
    #获取维数(行数),图像数(列数)
    dimens,nums=arr.shape[:2]
    print(arr.shape)
    #新建列表
    mean_arr=[]
    #遍历维数
    for i in range(dimens):
        aver=0
        #求和每个图像在该字段的值并平均
        aver=int(sum(arr[i,:])/float(nums))
        mean_arr.append(aver)
        #endfor
    return np.array(mean_arr)


def compute_eigenValues_eigenVectors(arr):
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
    eigenVectorsLabel = train_label.copy()
    eigenVectorsLabel = eigenVectorsLabel[:, idx]
    return eigenValues,np.dot(normArr,eigenVectors),eigenVectors, eigenVectorsLabel
 #   return eigenValues, eigenVectors.astype(np.float64)

def reconstruct(eigenVectors, eigenMat, meanImg, n):
    img = np.dot(eigenVectors[:,  0:n], eigenMat[0:n].reshape(n,1))+meanImg.reshape(meanImg.shape[0],1)
    img = img.reshape(57, 47)
    return img
def compute_img_mes(predictImgs, srcImgs):
    '''
    计算均方误差
    :param predictImgs: 行是维度，列是样本数
    :param srcImgs:
    :return:
    '''

    dim, nums = predictImgs.shape[:2]

    return np.sum(np.power(predictImgs-srcImgs, 2))/nums

def testMES():
    a = np.array([[1,2],
                  [3,4]])
    b = np.array([1,2])

    print(compute_img_mes(a,b))

def testTSNE():
    # # covertImg()
    prepareDataSet()

    eigenValues,eigenVectors, eigenMats = compute_eigenValues_eigenVectors(test_data.T)
    print(eigenVectors[0:100,:].shape)
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(eigenVectors[0:100,:].T)
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

    eig_vals, eig_vecs = np.linalg.eig(
        np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵

    sorted_indices = np.argsort(eig_vals)
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-targetDim - 1:-1]]  # 提取前k个特征向量
    return topk_eig_vecs

prepareDataSet()
resultVecs = LDA_dimensionality(train_data, train_label, 5)
print(resultVecs.shape)
#
# #avgImg = compute_mean_array(train_data.T)
# avgImg = np.mean(train_data, 0)
# eigenValues,eigenVectors, eigenMats = compute_eigenValues_eigenVectors(train_data.T)
# img = reconstruct(eigenVectors, eigenMats[61], avgImg, 150)
# print(compute_img_mes(img, train_data[1].reshape(57,47)))
# img1 = eigenVectors[:,1].reshape(57,47)
# # plt.subplot(1,5,1)
# plt.imshow(img, cmap='gray')
#
# plt.show()


