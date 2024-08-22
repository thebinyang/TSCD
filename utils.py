import numpy as np
import random
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def evaluation(pred_labels, gt):
    def kappa(confusion_matrix):
        """计算kappa值系数"""
        pe_rows = np.sum(confusion_matrix, axis=0)
        pe_cols = np.sum(confusion_matrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(confusion_matrix) / float(sum_total)
        return (po - pe) / (1 - pe)

    from sklearn.metrics import confusion_matrix
    confusionMatrixEstimated = confusion_matrix(y_true=gt.ravel(), y_pred=pred_labels, labels=[0, 1])
    TN, FP, FN, TP = confusionMatrixEstimated.ravel()
    print('TP=', TP, 'TN=', TN, 'FP=', FP, 'FN=', FN)

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    Kappa = kappa(confusionMatrixEstimated)
    print("test OA=", Accuracy, 'kpp=', Kappa, "Pre=", Precision, "Recall=", Recall, "F1=", F1_score)
    return Accuracy, Kappa

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy


def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros, require_AA_KPP=False):
    if False == require_AA_KPP:
        with torch.no_grad():
            available_label_idx = (train_samples_gt != 0).float()
            available_label_count = available_label_idx.sum()
            correct_prediction = torch.where(torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),available_label_idx, zeros).sum()
            OA = correct_prediction.cpu() / available_label_count
            return OA

class ChannelAttention(nn.Module):
    def __init__(self, ):
        super(ChannelAttention, self).__init__()
    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = torch.sigmoid(x1)
        return x1


class GKDM(nn.Module):
    def __init__(self, input_channels):
        super(GKDM, self).__init__()
        self.ca = ChannelAttention()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, input1, input2):
        input11 = self.fc1(input1)
        input22 = self.fc1(input2)
        input3 = input1 + input2
        input33 = self.fc1(input3)
        x1 = self.ca(input11)
        x2 = self.ca(input22)
        x3 = self.ca(input33)

        adj1 = torch.zeros(1, self.input_channels, 1, 1).to(device)
        adj2 = torch.zeros(1, self.input_channels, 1, 1).to(device)
        adj3 = torch.zeros(1, self.input_channels, 1, 1).to(device)
        for i in range(self.input_channels):
            if x1[:, i, :, :] >= x2[:, i, :, :]:
                if x1[:, i, :, :] >= x3[:, i, :, :]:
                    adj1[:, i, :, :] = 1
                else:
                    adj3[:, i, :, :] = 1
            if x1[:, i, :, :] <= x2[:, i, :, :]:
                if x2[:, i, :, :] >= x3[:, i, :, :]:
                    adj2[:, i, :, :] = 1
                else:
                    adj3[:, i, :, :] = 1
        x = adj1 * input1 + adj2 * input2 + adj3 * input3
        return x
