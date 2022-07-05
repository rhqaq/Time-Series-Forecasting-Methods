from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import warnings
import joblib
from multiprocessing import cpu_count
from tqdm import tqdm
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


def walk_forward_validation(data, n_test, cfg):
	# 定义一个给一套参数cfg打分的函数
	predictions = []
	train, test = data[:n_test], data[n_test:]

	history = [i for i in train]

	for x in range(len(test)):
		order = cfg[0]
		# print(order)
		model = ARIMA(history,order=order)
		model_fit = model.fit()
		yhat = model_fit.predict(len(history), len(history))
		predictions.append(yhat)
		history.append(test[x])
	error = mean_squared_error(test, predictions)
	print(predictions)
	return error


def score_model(data, n_test, cfg, debug=False):
	# 记录下一套参数，以及该参数下模型的得分
	key = str(cfg)
	# walk_forward_validation(data, n_test, cfg)
	if debug:
		error = walk_forward_validation(data, n_test, cfg)
	else:
		try:
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore')
				error = walk_forward_validation(data, n_test, cfg)
		except:
			error=None
	# if error is not None:
	# 	print(f'> Model{key} {error:.3f}')
	return key, error


def grid_search(data, cfg_list, n_test, parallel=False):
	# 把所有参数组合一一带入模型，并把所有参数组合及其对应模型得分记录下来，排序。
	if parallel:
		executor = joblib.Parallel(n_jobs=cpu_count(),
									backend='multiprocessing')
		tasks = (joblib.delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in tqdm(cfg_list)]
	scores = [r for r in scores if r[1] != None]
	scores.sort(key=lambda tup: tup[1])
	return scores

def sarima_config():
	# 造出自己预估的所有参数组合list
	cfg_list = []
	p_params = [0, 1, 2,3,4]
	d_params = [0,1,2]
	q_params = [0, 1, 2,3,4]

	for p in p_params:
		for d in d_params:
			for q in q_params:
				cfg = [(p,d,q)]
				cfg_list.append(cfg)
	return cfg_list


if __name__ == '__main__':
	# school = '东华'
	# df = pd.read_csv('2H{}流量变化.csv'.format(school))
	# print(df.head)
	# for index,rows in df.iterrows():
	# 	# if rows['Unnamed: 0'] == '2020-01-20 08:00:00':
	# 	# 	start = index
	# 	# if rows['Unnamed: 0'] == '2020-08-31 08:00:00':
	# 	# 	end = index
	# 	# 	break
	#
	# 	if rows['Unnamed: 0'] == '2020-09-16 08:00:00':
	# 		start = index
	# 	if rows['Unnamed: 0'] == '2021-01-22 08:00:00':
	# 		end = index
	# 		break

	# print(start)
	# print(end)
	# end = start+100
	# data = df.loc[start:end]['down_v6_average'].values
	# data = df['down_v4_average'].values
	area = 'su'
	df = pd.read_csv(r'D:\更新的代码\data\{}普通用户情感分析.csv'.format(area))
	print(df.head)
	flage = False
	for index,rows in df.iterrows():
		# if rows['Unnamed: 0'] == '2020-01-20 08:00:00':
		# 	start = index
		# if rows['Unnamed: 0'] == '2020-08-31 08:00:00':
		# 	end = index
		# 	break
		if rows['Unnamed: 0'] == '2020-03-01':
			start = index
			flage = True

		if flage:
			if rows['total_sum'] <= 5:
				rows['average_score'] = df.loc[index - 1]['average_score']


			# break
		# if rows['Unnamed: 0'] == '2021-01-22 08:00:00':
		# 	end = index
		# 	break
	data = df.loc[start:]['average_score'].values
	# for i in range(10):
	# 	print(adfuller(np.diff(np.diff(data,n=i))))
	test_slice = int(len(data)*0.9)+1
	valid_slice = int(len(data)*0.8)

	# result = seasonal_decompose(data[:test_slice], model='additive', period=12)
	#
	# residual = []
	# for number in result.resid:
	# 	if np.isnan(number):
	# 		residual.append(0)
	# 	else:
	# 		residual.append(number)
	# # print(residual)
	# regular_data = [data[i] - residual[i] for i in range(len(data[:test_slice]))]


	# result.plot()
	# plt.show()
	# print(data)

	# cfg_list = sarima_config()
	# scores = grid_search(data[:test_slice], cfg_list, valid_slice)
	# print(scores)
	# print('Done')
	# for cfg, error in scores[:5]: # 取出前5个最优的参数组合及对应的模型得分
	# 	print(cfg, error)
	# cfg, error = scores[0]
	# order = cfg[0]
	predictions_train = []
	predictions_v = []
	predictions_t = []
	# train, valid,test = regular_data[:valid_slice], regular_data[valid_slice:test_slice], data[test_slice:]
	train, valid, test = data[:valid_slice], data[valid_slice:test_slice], data[test_slice:]
	print(len(train))
	print(len(test))
	history = [i for i in train]
	# print(history)
	# history = history[0:24]
	# print(history)
	# order, sorder, trend = [(2, 1, 1), (0, 0, 0, 7), 'ct']

	order = (3,0,2)
	model = ARIMA(history,order=order)
	model_fit = model.fit()


	for x in range(len(test)):

		yhat = model_fit.predict(len(history), len(history))[0]
		# print(yhat)
		predictions_t.append(yhat)
		history.append(test[x])
		model = ARIMA(history,order=order)
		model_fit = model.fit()
	# fig_data = pd.concat([pd.DataFrame({'Original':data}), pd.DataFrame({'Regular Component':regular_data})], axis=1)
	print(predictions_t)
	fig_data = pd.concat([pd.DataFrame({'Original': data})], axis=1)
	fig_data['Prediction of Valid'] = ''
	fig_data['Prediction of Test'] = ''
	# np.save('{}arima4_train.npy'.format(school), np.array(predictions_train))
	# np.save('{}arima4_valid.npy'.format(school), np.array(predictions_v))
	# np.save('{}arima4 _test.npy'.format(school), np.array(predictions_t))
	for index,row in fig_data.iterrows():
		# if test_slice>int(index)>=valid_slice:
		# 	fig_data.loc[index,'Prediction of Valid'] = predictions_v[index-valid_slice]
		# 	# fig_data.loc[index, 'Regular Component'] = np.nan
		# else:
		# 	fig_data.loc[index, 'Prediction of Valid'] =np.nan

		if index>=test_slice:
			fig_data.loc[index, 'Prediction of Test'] = predictions_t[index - test_slice]
			# fig_data.loc[index, 'Regular Component'] = np.nan
		else:
			fig_data.loc[index, 'Prediction of Test'] = np.nan
	print(fig_data)
	err = [abs(predictions_t[i]-data[test_slice+i])/data[test_slice+i] for i in range(len(predictions_t))]
	err1 = [abs(data[test_slice + i-1] - data[test_slice + i]) / data[test_slice + i] for i in range(len(predictions_t))]
	err2 = [abs(data[test_slice + i - 7] - data[test_slice + i]) / data[test_slice + i] for i in range(len(predictions_t))]

	# print(np.mean(err2))
	print('mape')
	print(np.mean(err))
	print(np.mean(err1))
	print('mse')
	print(mean_squared_error(data[test_slice:],predictions_t))
	print(mean_squared_error(data[test_slice:], data[test_slice-1:-1]))
	# print(mean_squared_error(data[test_slice:], data[test_slice - 7:-7]))
	print('mae')
	print(mean_absolute_error(data[test_slice:],predictions_t))
	print(mean_absolute_error(data[test_slice:], data[test_slice-1:-1]))
	# print(mean_absolute_error(data[test_slice:], data[test_slice - 7:-7]))

	print(err)
	# plt.xlabel('The Absolute Value of Relative Error', fontsize=8, alpha=1)
	# plt.ylabel('CDF', fontsize=8, alpha=1)
	# kwargs = {'cumulative': True}
	# fig = sns.distplot(err, hist_kws=kwargs, kde_kws=kwargs)
	# fig.set_xlim([0, 1])

	print(len(fig_data))

	base_predict1 = [data[test_slice + i-1] for i in range(len(predictions_t))]
	base_predict2 = [data[test_slice + i - 7] for i in range(len(predictions_t))]
	# plt.figure(figsize=(12, 4))
	# plt.xlabel('Time(2H)', fontsize=8, alpha=1)
	# plt.ylabel('Traffic of ECNU v4', fontsize=8, alpha=1)
	# fig = sns.lineplot(data=fig_data['Original','Prediction of Test'][test_slice:])
	# line_fig = fig.get_figure()
	# plt.plot(base_predict1)
	# plt.plot(base_predict2)
	plt.plot(fig_data['Original'][test_slice:].values)
	plt.plot(fig_data['Prediction of Test'][test_slice:].values)
	# line_fig.savefig('华师大预测', dpi=400)

	plt.show()
	# error = mean_squared_error(test, predictions_t)
	# print(error)