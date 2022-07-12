import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from pandas import Series
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import random
from torch import optim
from Models import LSTM, MLP,Simple_LSTM,TimeLSTM
from sklearn.metrics import mean_squared_error , mean_absolute_error
import torch.nn as nn
import pandas as pd
from Getdata import sentiment_series,variable_series
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

# 转换成差分数据
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# 逆差分
def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]



def del_tensor_ele(arr,index):
    arr1 = arr[:,:,0:index]
    arr2 = arr[:,:,index+1:]
    return torch.cat((arr1,arr2),dim=2)

if __name__ == '__main__':
    seed_value = 2022
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.deterministic = True

    # <准备数据
    area = 'sui'
    df = pd.read_csv(r'D:\更新的代码\data\{}普通用户情感分析.csv'.format(area))
    flage = False
    for index, rows in df.iterrows():
        # if rows['Unnamed: 0'] == '2020-01-20 08:00:00':
        # 	start = index
        # if rows['Unnamed: 0'] == '2020-08-31 08:00:00':
        # 	end = index
        # 	break
        if rows['Unnamed: 0'] == '2020-03-01':
            start = index
            flage = True

        if flage:
            if rows['total_sum'] <=5:
                rows['average_score'] = df.loc[index-1]['average_score']
            # print(rows['Unnamed: 0'])

    # if rows['Unnamed: 0'] == '2021-01-22 08:00:00':
    # 	end = index
    # 	break
    # data_all = df.loc[start:]['average_score'].values
    # data_dif = difference(data_all).values

    params = {}

    best_mse_of_all = float("inf")
    # 准备数据>
    hidden_size=50
    lr=0.001
    time_step=7
    batch_size=8
    need_timeinter = True
    start_time = '2021-03-01'
    end_time = '2021-11-01'
    train_data,train_label,valid_data,valid_label,test_data,test_label = variable_series(time_step, area,start_time,end_time,need_timeinter)
    input_size = train_data.shape[2]-1
    # net = Simple_LSTM(input_size,hidden_size)
    net = TimeLSTM(input_size, hidden_size, 1, True)
    net = net.to(dev)

    loss_func = nn.MSELoss()
    opti = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    epoch_num = 1000

    # train_data,train_label,valid_data,valid_label,test_data,test_label = sentiment_series(data_dif,time_step)

    # print(train_data.shape)
    # print(valid_data.shape)
    # print(test_data.shape)
    # print(train_label.shape)
    # print(valid_label.shape)
    # print(test_label.shape)
    train_data_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=False)
    valid_data_loader = DataLoader(TensorDataset(valid_data, valid_label), batch_size=512, shuffle=False)
    test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=512, shuffle=False)
    best_mse = float("inf")


    for i in range(epoch_num):
        net.train()
        loss_ = AverageMeter()
        acc_ = AverageMeter()
        for data, label in train_data_loader:
            data, label = data.to(dev), label.to(dev)
            time_interval = data[:,:,13]
            data = del_tensor_ele(data,13)

            # preds = net(data)
            preds = net(data,time_interval) # T-lstm
            # print(preds.size())
            # print(label.size())
            loss = loss_func(preds, label)
            # weight = net.fc.weight.squeeze()
            # loss += c * torch.sum(torch.abs(weight))
            # print(loss.size())
            opti.zero_grad()
            loss.backward()
            opti.step()
            loss_.update(loss.item())

        # print("Epoch: %d, loss: %1.5f" % (i, loss_.sum))



        net.eval()  # eval下用验证集
        with torch.no_grad():
            label_list, preds_list = [], []
            for data, label in valid_data_loader:
                data, label = data.to(dev), label.to(dev)
                time_interval = data[:, :, 13]
                data = del_tensor_ele(data, 13)

                # preds = net(data)
                # preds = net(data)
                preds = net(data, time_interval)  # T-lstm
                label_list += label.cuda().data.cpu().numpy().tolist()
                preds_list += preds.cuda().data.cpu().numpy().tolist()
            a_s = mean_squared_error(label_list, preds_list)

        # print(a_s)
        if a_s < best_mse:
            best_mse = a_s
            best_epoch = i + 1  # 按照准确率作为最好的epoch
            torch.save(net.state_dict(), os.path.join('D:\时间序列预测\LSTM\savemodel', 'model{}.ckpt'.format(int(i + 1))))

        if i + 1 >= best_epoch + 30:
            print('Finish after epoch {}'.format(i + 1))
            # best_file_path = os.path.join('./savedmodel', '{}-best.ckpt'.format('mlp43'))
            # ori_path = os.path.join('./savedmodel', 'model-{}.ckpt'.format(best_epoch))
            # os.system('mv {} {}'.format(ori_path, best_file_path))  # 命令行将best移入best目录
            # os.system('rm -rf {}'.format(os.path.join('./savedmodel', 'model-*')))  # 一般模型清空
            break




    net.load_state_dict(
        torch.load(os.path.join('D:\时间序列预测\LSTM\savemodel', 'model{}.ckpt'.format(best_epoch))))

    # net.train()
    # for data, label in valid_data_loader:
    #     data, label = data.to(dev), label.to(dev)
    #     preds = net(data)
    #     # print(preds.size())
    #     # print(label.size())
    #     loss = loss_func(preds, label)
    #     # weight = net.fc.weight.squeeze()
    #     # loss += c * torch.sum(torch.abs(weight))
    #     # print(loss.size())
    #     opti.zero_grad()
    #     loss.backward()
    #     opti.step()
    #     loss_.update(loss.item())

    net.eval()
    mape_list = []
    with torch.no_grad():
        label_list, preds_list, prob = [], [], []
        for data, label in test_data_loader:
            data, label = data.to(dev), label.to(dev)
            time_interval = data[:,:,13]
            data = del_tensor_ele(data,13)

            # preds = net(data).squeeze()

            preds = net(data,time_interval).squeeze() # T-lstm
            # print(label)
            label_list += label.cuda().data.cpu().numpy().tolist()
            preds_list += preds.cuda().data.cpu().numpy().tolist()
            # mape_list.append((abs((label_list - preds_list) / label_list)))
    # print(preds_list)
    # print(label_list)
    # mape_list = [abs((preds_list[i]-label_list[i])/label_list[i]) for i in range(len(label_list))]
    # print('mape:{}'.format(np.mean(mape_list)))
    # print('mse:{}'.format(mean_squared_error(label_list,preds_list)))
    # print('mae:{}'.format(mean_absolute_error(label_list,preds_list)))
    #
    # # pre1_list = [abs((preds_list[i]-label_list[i])/label_list[i]) for i in range(len(label_list))]
    # # print('mape:{}'.format(np.mean(mape_list)))
    # # print('mse:{}'.format(mean_squared_error(label_list,preds_list)))
    # # print('mae:{}'.format(mean_absolute_error(label_list,preds_list)))
    # #
    # # mape_list = [abs((preds_list[i]-label_list[i])/label_list[i]) for i in range(len(label_list))]
    # # print('mape:{}'.format(np.mean(mape_list)))
    # # print('mse:{}'.format(mean_squared_error(label_list,preds_list)))
    # # print('mae:{}'.format(mean_absolute_error(label_list,preds_list)))
    #
    # # print(data)
    # # print(label_list)
    # plt.plot(label_list)
    # plt.plot(preds_list)
    # label_list = [inverse_difference(data_all,label_list[i],len(label_list) + 1 - i) for i in range(len(label_list))]
    # preds_list = [inverse_difference(data_all,preds_list[i],len(preds_list) + 1 - i) for i in range(len(preds_list))]
    mape_list = [abs((preds_list[i]-label_list[i])/label_list[i]) for i in range(len(label_list))]
    print('mape:{}'.format(np.mean(mape_list)))
    print('mse:{}'.format(mean_squared_error(label_list,preds_list)))
    print('mae:{}'.format(mean_absolute_error(label_list,preds_list)))
    plt.plot(label_list)
    plt.plot(preds_list)
    plt.show()