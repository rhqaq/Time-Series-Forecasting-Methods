import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def del_tensor_ele(arr,index):
    arr1 = arr[:,:,0:index]
    arr2 = arr[:,:,index+1:]
    return torch.cat((arr1,arr2),dim=2)

#定义create_inout_sequences函数，接收原始输入数据，并返回一个元组列表。
def create_inout_sequences(input_data, tw):
    data_seq = 0
    label_seq = 0
    L = len(input_data)
    input_size = input_data.shape[1]
    for i in range(L-tw):
        train_seq = input_data[i:i+tw].reshape(-1, input_size).unsqueeze(0)
        train_label = input_data[i+tw:i+tw+1,1].unsqueeze(0)#预测time_step之后的第一个数值,第1列是要预测的值
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
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # all_data_normalized = scaler.fit_transform(all_data.reshape(-1, 1))
    # all_data_normalized = torch.FloatTensor(all_data.reshape(-1, 1)).view(-1)
    all_data_normalized = torch.FloatTensor(all_data)

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

def variable_series(time_step,area,start_time,end_time,need_timeinter=False):
    df_liwc = pd.read_excel(r'../data/LIWC2015 Results (new_day_for_liwc_{}).xlsx'.format(area))
    # print(df_liwc.head)
    for index, rows in df_liwc.iterrows():
        if rows['Filename'] == '{}.txt'.format(start_time):
            start = index
        if rows['Filename'] == '{}.txt'.format(end_time):
            end = index
        df_liwc.loc[index, 'Filename'] = rows['Filename'][:-4]

    df_liwc = df_liwc.loc[start:end]
    print(df_liwc)
    df = pd.read_csv(r'../data/new_{}_lda_senti.csv'.format(area))
    flage = False
    for index, rows in df.iterrows():
        if rows['Unnamed: 0'] == '{}'.format(start_time):
            start = index

        if rows['Unnamed: 0'] == '{}'.format(end_time):
            end = index
            break

    df = df.loc[start:end]
    print(df)
    print(df.shape)
    df = pd.merge(df, df_liwc, how='left',left_on='Unnamed: 0',right_on='Filename')

    print(df.shape)
    for index, rows in df.iterrows():
        if rows['total_sum'] <= 5:
            df = df.drop(index=index)
    df.to_csv('test.csv')
    print(df.shape)
    data_all =  np.array(df)
    # time_interval = data_all[:,17]
    # print(time_interval)
    data_all = np.delete(data_all, [0,1,2,4,18,19], axis=1)
    print(data_all[0, :])
    data_all[:,0] = data_all[:,0] / data_all[:,0].max(axis=0)
    data_all[:, -17:] = data_all[:, -17:] / data_all[:, -17:].max(axis=0)
    if not need_timeinter:
        data_all = np.delete(data_all, 13, axis=1)
    print(data_all[:,:])
    data = sentiment_series(data_all[:,:].astype(np.float64),time_step)
    for dt in data:
        print(dt.shape)
    return data
    # create_inout_sequences()
if __name__ == '__main__':
    variable_series(4,'sui')