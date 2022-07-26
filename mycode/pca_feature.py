#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from numpy import *
import os
import numpy as np


suffix_list = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

def getImgAndLabel(src_dir):
    # 初始化返回结果
    imgs = []  # 存放图像
    labels = []  # 存放类别
    filenames = [] # 存放文件子地址
    filedirs = []
    # 获取子文件夹名
    # print(src_dir)
    catelist = os.listdir(src_dir)
    # print(catelist)
    # 遍历子文件夹
    for catename in catelist:
        # 设置子文件夹路径
        cate_dir = os.path.join(src_dir, catename)
        # 获取子文件名
        filelist = os.listdir(cate_dir)
        # 遍历所有文件名
        for filename in filelist:
            # 设置文件路径
            file_dir = os.path.join(cate_dir, filename)
            filedirs.append(file_dir)
            # 判断文件名是否为图片格式
            if not os.path.splitext(filename)[1] in suffix_list:
                print(file_dir, "is not an image")
                continue
            # endif
            # 读取灰度图
            # print(file_dir)
            imgs.append(cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE))
            # imgs.append(cv2.imread(file_dir, 0))
            # 读取相应类别
            labels.append(catename)
            filenames.append(filename)
    # endfor
    # endfor
    # print(imgs, labels,filenames,filedirs)
    return imgs, labels,filenames,filedirs
# end of getImgAndLabel

# 将图像数据变为一列
def convertImageToArray(img):
    img_arr = []
    height, width = img.shape[:2]
    # 遍历图像
    for i in range(height):
        img_arr.extend(img[i, :])
    # endfor
    return img_arr
# end of convertImageToArray

# 将每个图像变为一列
def convertImageToArrays(imgs):
    # 初始化数组
    arr = []
    # 遍历每个图像
    for img in imgs:
        arr.append(convertImageToArray(img)) # 按行添加
    # endfor
    return array(arr).T # 转化成列
# end of convertImageToArrays

# 计算均值数组
def compute_mean_array(arr):
    # 获取维数(行数),图像数(列数)————因为之前转置过
    dimens, nums = arr.shape[:2]
    # 新建列表
    mean_arr = []
    # 遍历维数
    for i in range(dimens):
        aver = 0
        # 求和每个图像在该字段的值并平均
        aver = int(sum(arr[i, :]) / float(nums))
        mean_arr.append(aver)
    # endfor
    return array(mean_arr)
# end of compute_mean_array

# 将数组转换为对应图像
def convert_array_to_image(arr, height, width):
    img = []
    for i in range(height):
        img.append(arr[i * width:i * width + width])
    # endfor
    return array(img)
# end of convert_array_to_image

# 计算图像和平均图像之间的差值
def compute_diff(arr, mean_arr):
    return arr - mean_arr
# end of compute_diff

# 计算每张图像和平均图像之间的差值
def compute_diffs(arr, mean_arr):
    diffs = []
    dimens, nums = arr.shape[:2]
    for i in range(nums):
        diffs.append(compute_diff(arr[:, i], mean_arr))
    # endfor
    return array(diffs).T
# end of compute_diffs

# 计算协方差矩阵的特征值和特征向量，按从大到小顺序排列
# arr是预处理图像的矩阵，每一列对应一个减去均值图像之后的图像
def compute_eigenValues_eigenVectors(arr):
    arr = array(arr) # 差值矩阵
    # 计算D'T * D
    temp = dot(arr.T, arr)
    eigenValues, eigenVectors = linalg.eig(temp) # 计算D'T * D的特征值和特征向量
    # 将特征值从大到小排序
    idx = np.argsort(-eigenValues)
    eigenValues = eigenValues[idx]
    # 特征向量按列排
    eigenVectors = eigenVectors[:, idx]
    return eigenValues, dot(arr, eigenVectors) #用差值矩阵D乘以D'T * D的特征向量
# end of compute_eigenValues_eigenVectors

# 计算图像在基变换后的坐标(权重)
def compute_weight(img, vec):
    return dot(img, vec)
# end of compute_weight

# 计算图像权重
def compute_weights(imgs, vec):
    dimens, nums = imgs.shape[:2]
    weights = []
    for i in range(nums):
        weights.append(compute_weight(imgs[:, i], vec))
    return array(weights)
# end of compute_weights

# 计算两个权重之间的欧式距离
def compute_euclidean_distance(wei1, wei2):
    # 判断两个向量的长度是否相等
    if not len(wei1) == len(wei2):
        print('长度不相等')
        os._exit(1)
    # endif
    sqDiffVector = wei1 - wei2
    sqDiffVector = sqDiffVector ** 2
    sqDistances = sqDiffVector.sum()
    distance = sqDistances ** 0.5
    return distance
# end of compute_euclidean_distance

# 计算待测图像与图像库中各图像权重的欧式距离
def compute_euclidean_distances(wei, wei_test):
    weightValues = []
    nums = wei.shape
    # print(nums)
    for i in range(nums[0]):
        weightValues.append(compute_euclidean_distance(wei[i], wei_test))
    # endfor
    return array(weightValues)
# end of compute_euclidean_distances

def select_components(arr_diff,n_components):
    arr=np.cov(arr_diff)
    [U, S, V] = np.linalg.svd(arr) #奇异值分解
    m=0
    n=S[0]
    i=0
    while(1):
        m+=S[i]
        n+=S[i+1]
        result=m/n
        if(result>n_components):
            break;
        i += 1
    return i+1
# end of select_components


def myPCA(src_dir,test_dir):
    '''
    :param src_dir:训练的图片
    :param test_dir:测试的图片
    :return:前五个最相似的图片和对应的欧式距离
    '''
    # 获取图片库路径
    # src_dir = os.path.join(os.getcwd(), "FaceDB_orl")
    src_dir = os.path.join(os.getcwd(), src_dir)
    # print(src_dir)

    # 获取图片以及对应类别
    imgs, labels, filenames, filedires = getImgAndLabel(src_dir)
    # for i in range(len(labels)):
    #     print(labels[i])
    # print(imgs)

    # 将图片转换为数组,10304x400
    arr = convertImageToArrays(imgs)
    # print("arr's shape : {}".format(arr.shape))

    # 计算均值图像
    mean_arr = compute_mean_array(arr)
    # print(mean_arr)
    # print("mean_arr's shape: {}".format(mean_arr.shape))
    # print("type(mean_arr): {}".format(type(mean_arr)))
    # 保存均值图片
    height = len(imgs[0])# 读取输入图像的长宽
    width = len(imgs[0][0])
    # print("height=",height,"width=",width)
    mean_img = convert_array_to_image(mean_arr, height, width)
    cv2.imwrite('../mycode/static/img/meanImage.png', mean_img)
    cv2.imwrite('../meanImage.png', mean_img)

    # 获取差值图像
    arr_diff = compute_diffs(arr, mean_arr)
    # print("arr_diff's shape: {}".format(arr_diff.shape))

    # 求协方差矩阵
    # arr=np.cov(arr_diff)
    # [U,S,V] = np.linalg.svd(arr)
    # # 降维后数据的保留信息
    # result0 = (S[0]) / (S[0] + S[1])
    # print(result0)  # 将数据从二维降到一维，保留了多少的信息
    # result1 = (S[0]+S[1])/(S[0]+S[1]+S[2])
    # print(result1) # 将数据从三维降到二维，保留了多少的信息
    # result2 = (S[0] + S[1] + S[2]) / (S[0] + S[1] + S[2] + S[3])
    # print(result2) # 将数据从四维降到三维，保留了多少的信息
    # result3 = (S[0] + S[1] + S[2] + S[3]) / (S[0] + S[1] + S[2] + S[3] + S[4])
    # print(result3)  # 将数据从五维降到四维，保留了多少的信息

    # k=select_components(arr_diff,n_components=0.9)
    # print("k=",k) #k=5

    # 计算特征值以及特征向量
    # eigenValues, eigenVectors = compute_eigenValues_eigenVectors(arr)
    eigenValues, eigenVectors = compute_eigenValues_eigenVectors(arr_diff)
    # print("eigenValues'shape : {}".format(eigenValues.shape))
    # print("eigenVectors'shape : {}".format(eigenVectors.shape))

    # 计算权重向量，此处假定使用特征值最大的前5个对应的特征向量作为基
    weights = compute_weights(arr_diff, eigenVectors[:, :5])
    # print("weights.shape : {}".format(weights.shape))
    # print(weights)

    # 读取测试图像，此处使用训练库中的一张
    # img_test = cv2.imread(r"FaceDB_orl/001/10.png", cv2.IMREAD_GRAYSCALE)
    img_test = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)
    arr_test = convertImageToArray(img_test)
    diff_test = compute_diff(arr_test, mean_arr) #测试脸与均值脸之差
    print("diff_test's shap : {}".format(diff_test.shape))
    diff_img = convert_array_to_image(diff_test,height,width)
    cv2.imwrite('../mycode/static/img/diffImage.png',diff_img)
    cv2.imwrite('../diffImage.png', diff_img)
    wei = compute_weight(diff_test, eigenVectors[:, :5])

    # print("test's weight : {}".format(wei))

    # 计算欧式距离（距离越小，越相似）
    # weights是某个图的三维向量
    # wei是测试图片的三维向量
    weightValues = compute_euclidean_distances(weights, wei)
    # print("weightValues.shape : {}".format(weightValues.shape))
    # 按从小到大排序
    sorted_id = sorted(range(len(weightValues)), key=lambda k: weightValues[k])

    sort_weightValues = []
    sort_filedires = []
    for i in range(5):
        sort_weightValues.append(weightValues[sorted_id[i]])
        sort_filedires.append(filedires[sorted_id[i]])
        print(weightValues[sorted_id[i]], filedires[sorted_id[i]])

    # 打印所有结果
    # for i in range(len(weightValues)):
    #     print(weightValues[i], labels[i],filenames[i])

    # 打印特征脸
    for i in range(len(eigenValues)):
        img=convert_array_to_image(eigenVectors[:,i],height, width)
        cv2.imwrite("./EigenFace/"+str(i)+".jpg" ,img)
        #break
        # print(eigenValues[i])
    #endfor
    # cv2.waitKey()
    # print("endl...")
    return sort_weightValues,sort_filedires


# endif
if __name__ == '__main__':
    # myPCA("FaceDB_orl","FaceDB_orl/001/10.png")
    myPCA(r"D:\!cy\! ZJSU\2 ZJSU_course\AI\AI_Dong\PCA\mycode\FaceDB_orl", r"D:\!cy\! ZJSU\2 ZJSU_course\AI\FaceDB_orl_test/1-10.png")
    # myPCA("FaceDB_orl", r"D:\!cy\! ZJSU\2 ZJSU_course\AI\AI_Dong\PCA\mycode\test1.bmp")
    # myPCA("FaceDB_orl", r"D:\!cy\! ZJSU\2 ZJSU_course\AI\AI_Dong\PCA\mycode\test2.bmp")