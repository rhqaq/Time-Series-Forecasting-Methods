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
from sklearn.metrics import mean_squared_error , mean_absolute_error,mean_absolute_percentage_error
import torch.nn as nn
import pandas as pd
from Getdata import sentiment_series,variable_series,del_tensor_ele
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-area', '--area', type=str, default='sui')

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


if __name__ == '__main__':
    seed_value = 2022
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

    args = parser.parse_args()
    args = args.__dict__

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.is_available())
    torch.backends.cudnn.deterministic = True

    # <准备数据
    area = args['area']
    print(area)

    params = {}
    best_mse_of_all = float("inf")
    # 准备数据>
    for hidden_size in [50,100,200]:
    # for hidden_size in [5]:
        for lr in [0.1,0.01,0.001]:
        # for lr in [0.1]:
            for time_step in [7]:
            # for time_step in [2]:
                for batch_size in tqdm([1,2,4,8]):
                # for batch_size in tqdm([1]):
                    need_timeinter = True
                    start_time = '2021-03-01'
                    end_time = '2021-11-01'
                    train_data, train_label, valid_data, valid_label, test_data, test_label = variable_series(time_step, area,
                                                                                                              start_time, end_time,
                                                                                                              need_timeinter)
                    input_size = 30
                    net = Simple_LSTM(input_size,hidden_size) #LSTM
                    net = TimeLSTM(input_size, hidden_size, 1, torch.cuda.is_available())
                    net = net.to(dev)

                    loss_func = nn.MSELoss()
                    opti = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
                    epoch_num = 1000


                    # train_data,train_label,valid_data,valid_label,test_data,test_label = variable_series(data_dif,time_step)
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
                            time_interval = data[:, :, 13]
                            data = del_tensor_ele(data, 13)
                            preds = net(data,time_interval)
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
                                preds = net(data,time_interval)
                                label_list += label.cuda().data.cpu().numpy().tolist()
                                preds_list += preds.cuda().data.cpu().numpy().tolist()
                            a_s = mean_squared_error(label_list, preds_list)

                        # print(a_s)
                        if a_s < best_mse:
                            best_mse = a_s
                            best_epoch = i + 1  # 按照准确率作为最好的epoch
                            # torch.save(net.state_dict(), os.path.join('D:\时间序列预测\LSTM\savemodel', 'model{}.ckpt'.format(int(i + 1))))

                        if i + 1 >= best_epoch + 30:
                            # print('Finish after epoch {}'.format(i + 1))
                            # best_file_path = os.path.join('./savedmodel', '{}-best.ckpt'.format('mlp43'))
                            # ori_path = os.path.join('./savedmodel', 'model-{}.ckpt'.format(best_epoch))
                            # os.system('mv {} {}'.format(ori_path, best_file_path))  # 命令行将best移入best目录
                            # os.system('rm -rf {}'.format(os.path.join('./savedmodel', 'model-*')))  # 一般模型清空
                            break

                    if best_mse<best_mse_of_all:
                        best_mse_of_all=best_mse
                        params['hidden_size'] = hidden_size
                        params['lr'] = lr
                        params['time_step'] = time_step
                        params['batch_size'] = batch_size
                        print(params)

    print('best params')
    print(params)
    with open('best_params_of_{}'.format(area), 'w', encoding='utf-8') as fp:
        json.dump(params, fp, indent=4, ensure_ascii=False)
