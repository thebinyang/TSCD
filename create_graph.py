import numpy as np
from collections import Counter

def split_data(ground_truth, class_num, radio, width, samples_type):
    Number_class = Counter(np.array(ground_truth))
    count = np.zeros(class_num+1)
    count[np.array(list(Number_class.keys())).astype(int)] = list(Number_class.values())
    count = count[1:]
    if samples_type == 'radio':
        train_count = list(np.around(count * radio).astype(int))
    else:
        train_count = [radio, radio]
    classes_index = []
    for i in range(class_num+1):  # with the background
        class_index = np.argwhere(ground_truth == i).reshape(-1)
        np.random.shuffle(class_index)
        classes_index.append(class_index)
    test_count = []
    train_index = []
    valid_index = []
    test_index = []
    for i in range(class_num):
        train_index.append(classes_index[i+1][:train_count[i]])
        valid_index.append(classes_index[i+1][(train_count[i]):2*train_count[i]])
        test_index.append(classes_index[i+1][2*train_count[i]:])
        test_count.append(len(test_index[i]))

    train_index = np.array(train_index, dtype=object)
    valid_index = np.array(valid_index, dtype=object)
    test_index = np.array(test_index, dtype=object)
    train_number = train_count[0] + train_count[1]
    test_number = test_count[0] + test_count[1]

    train_data_index = []
    valid_data_index = []
    test_data_index = []
    for i in range(train_index.shape[0]):
        a = train_index[i]
        for j in range(a.shape[0]):
            train_data_index.append(a[j])

    for i in range(valid_index.shape[0]):
        b = valid_index[i]
        for j in range(b.shape[0]):
            valid_data_index.append(b[j])

    for i in range(test_index.shape[0]):
        c = test_index[i]
        for j in range(c.shape[0]):
            test_data_index.append(c[j])
    print('Train count:', train_number)
    print('Valid count:', train_number)
    print('test count:', test_number)
    test_data_index = list(test_data_index)
    train_data_index = list(train_data_index)
    valid_data_index = list(valid_data_index)

    train_mn = np.zeros((len(train_data_index), 2), dtype=int)  ##总数=横坐标*width + 纵坐标
    for i in range(0, len(train_data_index)):
        p1 = train_data_index[i]
        train_mn[i, 0] = p1 // width
        train_mn[i, 1] = p1 % width

    val_mn = np.zeros((len(valid_data_index), 2), dtype=int)
    for i in range(0, len(valid_data_index)):
        p1 = valid_data_index[i]
        val_mn[i, 0] = p1 // width
        val_mn[i, 1] = p1 % width

    test_mn = np.zeros((len(test_data_index), 2), dtype=int)
    for i in range(0, len(test_data_index)):
        p1 = test_data_index[i]
        test_mn[i, 0] = p1 // width
        test_mn[i, 1] = p1 % width
    return train_data_index, valid_data_index, test_data_index, train_mn, val_mn, test_mn

def get_label(gt_reshape, train_index, val_index, test_index):
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_index)):
        train_samples_gt[train_index[i]] = gt_reshape[train_index[i]]

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_index)):
        test_samples_gt[test_index[i]] = gt_reshape[test_index[i]]

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_index)):
        val_samples_gt[val_index[i]] = gt_reshape[val_index[i]]

    return train_samples_gt, val_samples_gt, test_samples_gt



def GT_To_One_Hot(gt, class_count, height, width):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot

def get_label_mask(train_samples_gt, val_samples_gt, test_samples_gt, data_gt, class_num):
    
    height, width = data_gt.shape
    # train
    train_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num]) 
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [height * width, class_num])

    # val
    val_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height * width, class_num])

    # test
    test_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height * width, class_num])
    return train_label_mask, val_label_mask, test_label_mask