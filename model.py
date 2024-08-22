import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GKDM
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class TIGCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, band: int, temporal: int, A: torch.Tensor):
        super(TIGCN, self).__init__()
        self.A = A
        self.band = band
        self.temporal = temporal
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, input_dim*2))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)
        self.time_conv = nn.Conv2d(int(output_dim), int(output_dim), kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H):
        num_of_vertices, in_channels = H.shape
        H = H.reshape([num_of_vertices, int(in_channels/self.temporal), self.temporal])
        outputs = []
        for i in range(0, self.temporal):
            HH =H[:, :, i]
            HH = self.BN(HH)
            H_xx1 = self.GCN_liner_theta_1(HH)
            A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
            D_hat = self.A_to_D_inv(A)
            A_hat = torch.matmul(D_hat, torch.matmul(A, D_hat))
            output = self.Activition(torch.mm(A_hat, self.GCN_liner_out_1(HH)))
            outputs.append(output.unsqueeze(-1))

        conv_out = F.relu(torch.cat(outputs, dim=-1))
        a, b, c = conv_out.shape
        time_conv_output = self.time_conv(conv_out.unsqueeze(0).permute(0, 2, 1, 3))
        time_conv_output1 = torch.squeeze(time_conv_output, 0).permute(1, 0, 2).reshape(num_of_vertices, b*c)
        return F.relu(time_conv_output1)

class SSConv(nn.Module):   # 3D-CNN
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=1)
        self.BN = nn.BatchNorm3d(in_ch)
        self.Act11 = nn.LeakyReLU()

    def forward(self, input):
        input= self.BN(input)
        out = self.depth_conv(input)
        out = self.Act11(out)
        return out

class TSCD(nn.Module):
    def __init__(self, height: int, width: int, band: int, temporal: int, class_count: int, Q: torch.Tensor, A: torch.Tensor):
        super(TSCD, self).__init__()
        self.class_count = class_count
        self.band = band
        self.height = height
        self.width = width
        self.temporal = temporal
        self.Q = Q
        self.A = A
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        layers_count = 2

        # P3DC Sub-Network
        self.P3DC = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.P3DC.add_module('P3DC' + str(i), SSConv(band, 32, 3))
            else:
                self.P3DC.add_module('P3DC' + str(i), SSConv(32, 64, 3))

        # STIG Sub-Network
        self.STIG = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.STIG.add_module('STIG' + str(i), TIGCN(band, 64, band, temporal, self.A))
            else:
                self.STIG.add_module('STIG' + str(i), TIGCN(64, 64, band, temporal, self.A))

        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(256, self.class_count))
        self.Softmax_linear1 = nn.Sequential(nn.Linear(512, self.class_count))
        self.d1conv = nn.Conv1d(in_channels=64*temporal, out_channels=256, kernel_size=1)
        self.d2conv = nn.Conv2d(in_channels=64*temporal, out_channels=256, kernel_size=1)
        self.GKDM = GKDM(input_channels=256)
    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        y = x.view(self.height, self.width, self.temporal, self.band)  ###(h, w, 10, 6)
        output1 = y.permute([2, 3, 0, 1])   ###(10, 6, h, w)

        CNN_result = self.P3DC(torch.unsqueeze(output1.permute([1, 0, 2, 3]), 0))
        CNN_result1 = torch.squeeze(CNN_result, 0).permute([2, 3, 1, 0]).reshape([self.height, self.width, -1]).permute([2, 0, 1])  ###(640, h, w)
        CNN_result2 = self.d2conv(CNN_result1.unsqueeze(0)) ###(1, 256, h, w)

        clean_x_flatten = x.reshape([self.height * self.width, -1])  ##(h*w, 60)
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)

        H = superpixels_flatten
        for i in range(len(self.STIG)):
            H = self.STIG[i](H)

        H1 = self.d1conv(torch.unsqueeze(H, 0).permute(0, 2, 1)).permute(0, 2, 1).squeeze(0)
        GCN_result = torch.matmul(self.Q, H1)  # superpixel to pixel

        # feature selection
        GCN_result1 = torch.unsqueeze(GCN_result.reshape([self.height, self.width, -1]).permute([2, 0, 1]), 0)
        Y = self.GKDM(GCN_result1, CNN_result2)
        Y = torch.squeeze(Y, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])
        Y = F.softmax(self.Softmax_linear(Y), -1)
        return Y

