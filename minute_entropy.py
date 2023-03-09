import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

dir_main = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/'
dir_data_input = dir_main + 'result/'
dir_data_output = dir_main + 'result/'

minute_std = pd.read_hdf(dir_data_input + 'minute_std.h5', key='df')
minute_std = pd.DataFrame(minute_std.values, index=minute_std.index, columns=minute_std.columns.astype(int))
date_list = list(minute_std.columns)
stock_code = list(minute_std.index)

# ==============
# Get date range
last_dates = []
first_dates = []
date_range = {}   # At last dates of each month, look back 20 days

for i in range(len(date_list) - 1):
    if str(date_list[i])[:6] != str(date_list[i+1])[:6]:
        last_dates.append(date_list[i])
        first_dates.append(date_list[i+1])
        if i > 20:
            date_range[date_list[i]] = []
            for j in range(i-19, i+1):
                date_range[date_list[i]].append(date_list[j])

# ===========
# Data filter
dir_data_input = dir_main + 'data/'
df1 = pd.read_hdf(dir_data_input + 'trading_day.h5', key='df')
df2 = pd.read_hdf(dir_data_input + 'ST.h5', key='df')
df3 = pd.read_hdf(dir_data_input + 'trading_state.h5', key='df')
dates = list(df2.columns)[2:]
df_filter = pd.DataFrame(index=range(len(df1)), columns=dates, data=False)
df_filter[df1[dates] < 60] = True
df_filter[df2[dates] == 1] = True
df_filter[df3[dates] != '交易'] = True
IDs = [x[:6] for x in list(df1['ID'])]
df_filter = df_filter.rename(index = dict(zip(range(len(df1)), IDs)))
minute_std[df_filter[dates] == True] = np.nan

# =============
# Calculate UTD
def calc_UTD():
    UTD = pd.DataFrame(columns=last_dates[1:], index=stock_code)
    
    for stock in stock_code:
        print(stock)
        std_data = minute_std.loc[stock,:]
        
        for date in last_dates[1:]:
            std_20 = std_data[date_range[date]]
            if std_20.count() > 15:
                Std = std_20.std()
                Mean = std_20.mean()
                UTD.loc[stock, date] = Std / Mean
    return UTD

UTD20 = calc_UTD()
UTD20.to_hdf(dir_data_output + 'UTD.h5', key='df')

# ===================
# UTD size neutrality
dir_data_input = dir_main + 'result/data/'
capital = pd.read_hdf(dir_data_input + 'capital.h5', key='df')
close = pd.read_hdf(dir_data_input + 'close_before.h5', key='df')
IDs = [x[:6] for x in list(close['ID'])]
capital = capital.rename(dict(zip(range(len(capital)), IDs)))
close = close.rename(dict(zip(range(len(close)), IDs)))
UTD20 = UTD20.loc[IDs, :]
model = linear_model.LinearRegression()

UTD_deSize = pd.DataFrame(columns=last_dates[1:], index=IDs)
for date in last_dates[1:]:
    y = UTD20[date]
    cap = capital[date]
    price = close[date]
    size = cap * price
    x = np.log(size.replace(0.0, np.nan))
    temp = x.isna() | y.isna()
    x = x[temp == False].values.reshape(-1, 1)
    y = y[temp == False]
    model.fit(x, y)
    y_predict = model.predict(x)
    UTD_deSize[date][temp == False] = y - y_predict

UTD_deSize.to_hdf(dir_data_output + 'UTD_deSize.h5', key='df')

# ========================================
# A simpler way to calculate factor return
dir_data_input = dir_main + 'result/'
ret = pd.read_hdf(dir_data_input + 'ret.h5', key='df')
ret = pd.DataFrame(ret.values, index=IDs, columns=ret.columns)

def calc_ret_simple(factor, num):
    data = pd.read_hdf(dir_data_input + factor + '.h5', key='df')
    all_dates = list(data.columns)[:-1]
    monthly_ret = pd.DataFrame(index=all_dates, columns=range(1, num+1))
    
    for date in all_dates:
        rtn = ret[date]
        notnan_index = ret[date].notnull() & data[date].notnull()
        sorted_index = data[date][notnan_index].sort_values().index
        groups = np.array_split(sorted_index, num)
        
        for i in range(num):
            group_index = groups[i]
            monthly_ret.loc[date, i+1] = rtn[group_index].mean()

    monthly_ret.to_excel(dir_data_output + factor + '.xlsx')
    return monthly_ret

calc_ret_simple('UTD_deSize', 10)

# ============================
# Calculate entropy factor H30
entropy30 = pd.read_hdf(dir_data_input + 'entropy30.h5', key='df')
entropy30 = pd.DataFrame(entropy30.values, index=entropy30.index, columns=entropy30.columns.astype(int))
entropy30[df_filter[dates] == True] = np.nan

def calc_entropy(H):
    entropy = pd.DataFrame(columns=last_dates[1:], index=stock_code)
    
    for stock in stock_code:
        print(stock)
        H_data = H.loc[stock,:]
        
        for date in last_dates[1:]:
            H_20 = H_data[date_range[date]]
            if H_20.count() > 15:
                Std = H_20.std()
                Mean = H_20.mean()
                entropy.loc[stock, date] = Std / Mean
    return entropy

H30 = calc_entropy(entropy30)
H30.to_hdf(dir_data_output + 'H30.h5', key='df')

# ===================
# H30 size neutrality
dir_data_input = dir_main + 'result/data/'
capital = pd.read_hdf(dir_data_input + 'capital.h5', key='df')
close = pd.read_hdf(dir_data_input + 'close_before.h5', key='df')
IDs = [x[:6] for x in list(close['ID'])]
capital = capital.rename(dict(zip(range(len(capital)), IDs)))
close = close.rename(dict(zip(range(len(close)), IDs)))
H30 = H30.loc[IDs, :]
model = linear_model.LinearRegression()

H30_deSize = pd.DataFrame(columns=last_dates[1:], index=IDs)
for date in last_dates[1:]:
    y = H30[date]
    cap = capital[date]
    price = close[date]
    size = cap * price
    x = np.log(size.replace(0.0, np.nan))
    temp = x.isna() | y.isna()
    x = x[temp == False].values.reshape(-1, 1)
    y = y[temp == False]
    model.fit(x, y)
    y_predict = model.predict(x)
    H30_deSize[date][temp == False] = y - y_predict

H30_deSize.to_hdf(dir_data_output + 'H30_deSize.h5', key='df')
calc_ret_simple('H30_deSize', 10)

