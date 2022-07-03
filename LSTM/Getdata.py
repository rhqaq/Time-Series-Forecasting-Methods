import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

#定义create_inout_sequences函数，接收原始输入数据，并返回一个元组列表。
def create_inout_sequences(input_data, tw):
    data_seq = 0
    label_seq = 0
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw].reshape(-1, 1).unsqueeze(0)
        train_label = input_data[i+tw:i+tw+1].unsqueeze(0)#预测time_step之后的第一个数值
        # inout_seq.append((train_seq, train_label))#inout_seq内的数据不断更新，但是总量只有tw+1个

        if not torch.is_tensor(data_seq):
            data_seq = train_seq
            label_seq = train_label
        else:
            data_seq = torch.cat((data_seq, train_seq), 0)
            label_seq = torch.cat((label_seq, train_label), 0)
        # data_seq.append(train_seq)
        # label_seq.append(train_label)
    return data_seq,label_seq.squeeze()

def sentiment_series(df,time_step):
    all_data = df
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # all_data_normalized = scaler.fit_transform(all_data.reshape(-1, 1))
    all_data_normalized = torch.FloatTensor(all_data.reshape(-1, 1)).view(-1)


    all_data_normalized,all_label_normalized = create_inout_sequences(all_data_normalized, time_step)

    test_slice = int(len(all_data_normalized) * 0.9)
    valid_slice = int(len(all_data_normalized) * 0.8)

    train_data = all_data_normalized[:valid_slice]
    valid_data = all_data_normalized[valid_slice:test_slice]
    test_data = all_data_normalized[test_slice:]
    train_label = all_label_normalized[:valid_slice]
    valid_label = all_label_normalized[valid_slice:test_slice]
    test_label = all_label_normalized[test_slice:]
    return train_data,train_label,valid_data,valid_label,test_data,test_label