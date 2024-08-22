import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import LDA_SLIC
import matplotlib.pyplot as plt
from model import TSCD
from utils import compute_loss, evaluate_performance, seed_torch, evaluation
from create_graph import get_label, get_label_mask, GT_To_One_Hot, split_data
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
curr_seed = 200
seed_torch(200)

SC = 3
data_mat = sio.loadmat(r'./Data/scene{}/data.mat'.format(SC))
img = data_mat['data']  ###(10 6 400 400)
temporal, bands, height, width = img.shape
channel = temporal * bands
print('height', height, 'width', width, 'bands', bands, 'temporal', temporal)
img = np.reshape(img, (channel, height, width))  ##(60 400 400)
data = np.transpose(img, (1, 2, 0))
sname = 'scene{}'.format(SC)
print(sname)

gt1 = data_mat['allmap1']
gt1[gt1 == 1] = 3  # 2:unchanged，1:changed，0:unmarked
gt1[gt1 == 0] = 1
gt = gt1 - 1
gt_reshape = np.reshape(gt, [-1])
background = np.argwhere(gt == 0)
# parameter setting
samples_type = 'radio'
train_ratio = 0.01
class_count = 2
learning_rate = 3e-4
max_epoch = 200
dataset_name = "scene"
superpixel_scale = 150

data = np.reshape(data, [height * width, channel])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [height, width, channel])

train_data_index, val_data_index, test_data_index, train_pos, valid_pos, test_pos = split_data(gt_reshape, class_count, train_ratio, width, samples_type)
train_samples_gt, val_samples_gt, test_samples_gt = get_label(gt_reshape, train_data_index, val_data_index, test_data_index)
train_label_mask, val_label_mask, test_label_mask = get_label_mask(train_samples_gt, val_samples_gt, test_samples_gt, gt, class_count)

train_gt = np.reshape(train_samples_gt, [height, width])
val_gt = np.reshape(val_samples_gt, [height, width])
test_gt = np.reshape(test_samples_gt, [height, width])

train_samples_gt_onehot = GT_To_One_Hot(train_gt, class_count, height, width)
val_samples_gt_onehot = GT_To_One_Hot(val_gt, class_count, height, width)
test_samples_gt_onehot = GT_To_One_Hot(test_gt, class_count, height, width)

train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)
test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)

ls = LDA_SLIC.LDA_SLIC(data, train_gt, class_count - 1)
Q, S, A, Seg = ls.simple_superpixel_no_LDA(scale=superpixel_scale)
Q = torch.from_numpy(Q).to(device)
A = torch.from_numpy(A).to(device)

train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

net_input = np.array(data, np.float32)
net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)

net = TSCD(height, width, bands, temporal, class_count, Q, A)
net.to(device)

zeros = torch.zeros([height * width]).to(device).float()

# train
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
best_accuracy=0
net.train()

for i in range(max_epoch + 1):
    optimizer.zero_grad()
    output = net(net_input)
    loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        net.eval()
        output = net(net_input)
        trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
        trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot, zeros)

        valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
        valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot, zeros)
        if torch.isnan(trainloss):
            break
        if best_accuracy < valOA:
            best_accuracy = valOA
            torch.save(net.state_dict(), './Model/best_model_scene.pt')

    torch.cuda.empty_cache()
    net.train()
    if i % 10 == 0:
        print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA))
print("\n\n====================training done. starting evaluation...========================\n")

torch.cuda.empty_cache()
with torch.no_grad():
    net.load_state_dict(torch.load("./Model/best_model_scene.pt"))
    net.eval()
    output = net(net_input)
    testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
    output = torch.argmax(output, 1).cpu().numpy()
    test_gt[test_gt == 0] = 3
    test_samples_gt1 = test_gt-1
    testOA, kappa = evaluation(output, test_samples_gt1)

    # predictiom map
    classification_map = np.reshape(output, [height, width])
    classification_map = classification_map + 1
    classification_map[classification_map == 1] = 0
    classification_map[background[:, 0], background[:, 1]] = 1
    plt.imsave("./result_{}_{}.png".format(sname, testOA), classification_map, cmap='gray')

torch.cuda.empty_cache()
del net
