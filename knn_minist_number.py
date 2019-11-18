"""
K-近邻算法实现手写数据集识别系统(手写数据集Minist)
"""
# 加载库
import numpy as np
import pandas as pd
import operator

# 获取数据集
def get_digit():
    print('test_images...')
    test_images = pd.read_csv("E:\\PYTHONproject\\BAG\\机器学习数据\\Minist_test_data.csv")
    print('test_labels...')
    test_labels = pd.read_csv("E:\\PYTHONproject\\BAG\\机器学习数据\\Minist_test_labels.csv")
    print(test_images.shape, test_labels.shape)
    # 训练集作为总集
    return np.array(test_images), np.array(test_labels)

# 数据归一化方式1
def No1_norm_dataset(Dataset):
    """
    归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 newdataset
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。
        该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算数据集的最大、最小值
    dataset_min = Dataset.min()  # 默认情况下求取整体的最小值
    dataset_max = Dataset.max()  # 默认情况下求取整体的最大值
    print(dataset_min, dataset_max)
    # 获得Xmax-Xmin
    ranges = dataset_max - dataset_min
    # 获得X-Xmin
    # np.tile:将dataset_min作为整体,将其复制成大小为(Dataset.shape[0], 1)
    newdataset = Dataset - np.tile(dataset_min, (Dataset.shape[0], Dataset.shape[1]))
    print(newdataset.shape)
    # 实现归一化
    newdataset = newdataset/np.tile(ranges, (Dataset.shape[0], Dataset.shape[1]))
    print(newdataset.shape)
    return newdataset

# 数据归一化方式2
def No2_norm_dataset(Dataset):

    return Dataset

# 欧式距离计算方式
def Euclidean_distance(test_x, dataset):
    """
    :param test_x: 测试数据
    :param dataset: 整体数据集
    :return: 根据测试数据和整体数据集算得的欧式距离值
    """
    # 获取数据集的总量
    dataset_size = dataset.shape[0]
    # 距离度量 度量公式为欧氏距离
    diff = np.tile(test_x, (dataset_size, 1)) - dataset  # 第一步:计算各自差值
    sqdiff = diff**2  # 第二步:各自差值的平方
    distance = sqdiff.sum(axis=1)  # 第三步:计算每一行的和值
    distance = distance ** 0.5  # 第四步:计算距离的开方值,得到真实欧式距离值
    return distance

# 曼哈顿距离计算方式
def Manhattan_distance(test_x, dataset):
    """
    :param test_x: 测试数据
    :param dataset: 整体数据集
    :return: 根据测试数据和整体数据集算得的曼哈顿距离值
    """
    # 获取数据集的总量
    dataset_size = dataset.shape[0]
    # 距离度量 度量公式为曼哈顿距离
    diff = np.tile(test_x, (dataset_size, 1)) - dataset  # 第一步:计算各自差值
    absdiff = abs(diff)  # 第二步:计算各自差值的绝对值
    distance = absdiff.sum(axis=1)  # 第三步:计算每一行的和值(即可获得距离值)
    return distance

# 切比雪夫距离计算方式
def Chebyshev_distance(test_x, dataset):
    """
    :param test_x: 测试数据
    :param dataset: 整体数据集
    :return: 根据测试数据和整体数据集算得的切比雪夫距离值
    """
    # 获取数据集的总量
    dataset_size = dataset.shape[0]
    # 距离度量 度量公式为曼哈顿距离
    diff = np.tile(test_x, (dataset_size, 1)) - dataset  # 第一步:计算各自差值
    absdiff = abs(diff)  # 第二步:计算各自差值的绝对值
    distance = absdiff.max(axis=1)  # 第三步:计算每一行的最大值(即可获得距离值)
    return distance

# 余弦距离计算方式
def cosine_distance(test_x, dataset):
    """
    :param test_x: 测试数据
    :param dataset: 整体数据集
    :return: 根据测试数据和整体数据集算得的余弦距离值
    (用1-cosD作为返回值,cosD越大,1-cosD越小,因为调用区排序需要从小到大排序)
    规则:计算出来的值越接近1,则说明两者之间越相似
    """
    cosD = np.dot(test_x, dataset.T) / (np.linalg.norm(test_x) * np.linalg.norm(dataset, axis=1))
    distance = 1 - cosD
    return distance

# knn分类预测
def knn_classify(test_x, dataset, labels, k, cal_function='Euclidean'):
    """
    :param test_x: 测试数据
    :param dataset:整体数据集
    :param labels: 整体数据集标签
    :param k: 整体前k个最短距离
    :param cal_function:距离计算方式,默认是Euclidean(欧式距离计算法)
    :return: class_label预测的类别
    """
    # 距离函数定义字典
    distance_function = {'Euclidean': Euclidean_distance,  # 欧氏距离
                         'Manhattan': Manhattan_distance,  # 曼哈顿距离
                         'Chebyshev': Chebyshev_distance,  # 切比雪夫距离
                         'cosine': cosine_distance}  # 余弦距离
    # 判断距离计算方式是否是指定的
    if cal_function in distance_function.keys():
        print('cal_function:', cal_function)
        nowdistance = distance_function[cal_function](test_x, dataset)
    else:  # 距离计算方式不对,返回指定值100(指代说明错误率将高达100%)
        return 100
    # 将距离从小到大排序
    sortedDistance = nowdistance.argsort()  # 返回排序后的数据索引值
    # 记录前k个距离对应的类别出现次数(使用字典计数)
    labelsCount = {}
    for i in range(k):
        nowlabel = labels[sortedDistance[i], 0]  # 按照排序结果取出标签值
        # 按照类别计数,labelsCount.get(nowlabel, 0),没有就返回默认值0
        labelsCount[nowlabel] = labelsCount.get(nowlabel, 0) + 1
    # 对labelsCount进行排序(按照出现次数)
    sortedlabelsCount = sorted(labelsCount.items(), key=operator.itemgetter(1), reverse=True)
    class_label = sortedlabelsCount[0][0]  # 出现次数最多的标签值
    return class_label

# 主函数
def main():
    # 获取数据(数据集以及对应的分类标签)
    print('get_digit...')
    dataset, datasetLabels = get_digit()
    # 数据归一化方式1
    normdataset = No1_norm_dataset(dataset)
    print(datasetLabels[0:3])
    print(datasetLabels[1, 0])
    test_length = int(0.1*normdataset.shape[0])
    # 进行测试集的预测
    k = 5  # 取前k个最短距离值
    errcount = 0.0  # 计算误差率
    for i in range(test_length):
        print('predict:', i+1)
        classifier = knn_classify(normdataset[i, :], normdataset[test_length:, :],
                                  datasetLabels[test_length:], k, cal_function='Manhattan')
        print('the predict class is %d,the real class is %d' % (classifier, datasetLabels[i, 0]))
        if classifier != datasetLabels[i, 0]:
            errcount = errcount + 1
    print('Total:%d,errcount:%d,err percent:%f%%' % (test_length, errcount, 100*errcount/test_length))

if __name__ == '__main__':
    main()
