import scipy.io as sio
import os
from param import *
from utils import *
from utils import gen_dataset_from_dict
import spectral as spy
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
spy.settings.WX_GL_DEPTH_SIZE = 100

CLASSES = []
NEW_DATA_PATH = os.path.join(os.getcwd(), "new data")
FILE_PATH = './Datasets/DPaviaU-DPaviaC.mat'
# K = 50  # param for pca
BUFFER_SIZE = 1000
BATCH_SIZE = 1

if not os.path.exists(NEW_DATA_PATH):
    os.mkdir(NEW_DATA_PATH)


def load_data(filePath, dataName, GroundTruth):
    file = sio.loadmat(filePath)
    data = file[dataName]
    ground_truth = file[GroundTruth]
    return data, ground_truth


data_source, gt_source = load_data(FILE_PATH, 'DataCube1', 'gt1')
data_target, gt_target = load_data(FILE_PATH, 'DataCube2', 'gt2')

data_source = normalized(data_source)
data_target = normalized(data_target)
# data_source = pca(data_source, K)
# data_target = pca(data_target, K)

for _ in range(int(CLASSES_NUM)):
    CLASSES.append([])


def DataFilter(data, ground_truth):
    """生成数据, 过滤背景像元
       ground_truth转为one_hot"""
    NEW_DATA = []
    NEW_GT = []
    depth = 6
    for i in range(int(data.shape[0])):
        for j in range(int(data.shape[1])):
            curve = [data[i][j].astype('float32'), ]
            gt = ground_truth[i, j].astype('float32')
            if gt != 0:
                # print(gt, ':', LABELS[int(gt) - 1], ':', curve)
                gt = tf.one_hot(gt-1, depth=depth)
                print(gt)
                NEW_DATA.append(curve)
                NEW_GT.append(gt)
    return NEW_DATA, NEW_GT


def SaveData(data, gt, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    data_dict = {'data': data, 'gt': gt}
    sio.savemat(os.path.join(path, name + '.mat'), data_dict)


new_data_source, new_gt_source = DataFilter(data_source, gt_source)
new_data_target, new_gt_target = DataFilter(data_target, gt_target)

new_data_target, validation_data, new_gt_target, validation_gt = train_test_split(new_data_target,
                                                                                  new_gt_target,
                                                                                  test_size=0.1,
                                                                                  random_state=42)

SaveData(new_data_source, new_gt_source, NEW_DATA_PATH, 'Source')
SaveData(new_data_target, new_gt_target, NEW_DATA_PATH, 'Target')

