from sklearn.svm import SVR
import random
import numpy as np
import os
import pandas as pd
from Getdata import sentiment_series
from pandas import Series
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error , mean_absolute_error
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


def evaluate_model(preds_list,label_list):
    mape_list = [abs((preds_list[i]-label_list[i])/label_list[i]) for i in range(len(label_list))]
    print('mape:{}'.format(np.mean(mape_list)))
    print('mse:{}'.format(mean_squared_error(label_list,preds_list)))
    print('mae:{}'.format(mean_absolute_error(label_list,preds_list)))

if __name__ == '__main__':
    seed_value = 2022
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    # <准备数据
    area = 'su'
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
            print(rows['Unnamed: 0'])

    # if rows['Unnamed: 0'] == '2021-01-22 08:00:00':
    # 	end = index
    # 	break
    data_all = df.loc[start:]['average_score'].values
    data_dif = difference(data_all).values
    train_data, train_label, valid_data, valid_label, test_data, test_label = sentiment_series(data_dif, 5)
    train_data, train_label, valid_data, valid_label, test_data, test_label = train_data.squeeze().numpy().tolist(), train_label.numpy().tolist(), valid_data.squeeze().numpy().tolist(), valid_label.numpy().tolist(), test_data.squeeze().numpy().tolist(), test_label.numpy().tolist()
    train_data, train_label = train_data+valid_data, train_label+valid_label

    parameters = {'kernel': ['rbf'], 'gamma': np.logspace(-5, 0, num=6, base=2.0),
                  'C': np.logspace(-3, 3, num=7, base=2.0)}

    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, parameters, cv=5, n_jobs=4, scoring='neg_mean_squared_error')

    print(train_data[0])
    print(train_label[0])
    # grid_search.fit(train_data, train_label)
    grid_search.fit(train_data, train_label)
    y_hat = grid_search.predict(test_data)


    y_hat = [inverse_difference(data_all,y_hat[i],len(y_hat) + 1 - i) for i in range(len(y_hat))]
    test_label = [inverse_difference(data_all,test_label[i],len(y_hat) + 1 - i) for i in range(len(test_label))]
    evaluate_model(y_hat, test_label)
    plt.plot(y_hat)
    plt.plot(test_label)
    plt.show()