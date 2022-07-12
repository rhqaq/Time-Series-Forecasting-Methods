import matplotlib.pyplot as plt
import pandas as pd

for area in ['xi','sui','su','bei','yue']:
# area = 'xi'
    df = pd.read_csv(r'D:\更新的代码\data\{}普通用户情感分析.csv'.format(area))
    print(df.head)
    flage = False
    for index, rows in df.iterrows():
        # if rows['Unnamed: 0'] == '2020-01-20 08:00:00':
        # 	start = index
        # if rows['Unnamed: 0'] == '2020-08-31 08:00:00':
        # 	end = index
        # 	break
        if rows['Unnamed: 0'] == '2020-06-01':
            start = index
            flage = True

        if flage:
            if rows['total_sum'] <= 5:
                # rows['average_score'] = df.loc[index - 1]['average_score']
                df.loc[index, 'average_score'] = df.loc[index - 1]['average_score']

        # break
        if rows['Unnamed: 0'] == '2020-10-31':
            end = index
            break
    data = df.loc[start:end]['average_score'].values
    # x = df.loc[start:]['Unnamed: 0'].values
    plt.plot(data,label=area)
    plt.show()
