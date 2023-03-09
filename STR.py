import numpy as np
import pandas as pd
from sklearn import linear_model

dir_main = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/'
dir_data_input = dir_main + 'data/'

# ===============
# Data preprocess
dir_data_output = dir_main + 'result/data/'

capital = pd.read_hdf(dir_data_input + 'capital.h5', key='df')
open_after = pd.read_hdf(dir_data_input + 'open_after.h5', key='df')
close_before = pd.read_hdf(dir_data_input + 'close_before.h5', key='df')
close_after = pd.read_hdf(dir_data_input + 'close_after.h5', key='df')
turnover = pd.read_hdf(dir_data_input + 'turnover.h5', key='df')

df1 = pd.read_hdf(dir_data_input + 'trading_day.h5', key='df')
df2 = pd.read_hdf(dir_data_input + 'ST.h5', key='df')
df3 = pd.read_hdf(dir_data_input + 'trading_state.h5', key='df')
dates = list(df2.columns)[2:]
df_filter = pd.DataFrame(index=range(len(df1)), columns=dates, data=False)
df_filter[df1[dates] < 60] = True
df_filter[df2[dates] == 1] = True
df_filter[df3[dates] != '交易'] = True

capital[df_filter[dates] == True] = np.nan
open_after[df_filter[dates] == True] = np.nan
close_before[df_filter[dates] == True] = np.nan
close_after[df_filter[dates] == True] = np.nan
turnover[df_filter[dates] == True] = np.nan

capital.to_hdf(dir_data_output + 'capital.h5', key='df')
open_after.to_hdf(dir_data_output + 'open_after.h5', key='df')
close_before.to_hdf(dir_data_output + 'close_before.h5', key='df')
close_after.to_hdf(dir_data_output + 'close_after.h5', key='df')
turnover.to_hdf(dir_data_output + 'turnover.h5', key='df')

# ================
# Calculate factor
dir_data_input = dir_main + 'result/data/'
dir_data_output = dir_main + 'result/'

turnover = pd.read_hdf(dir_data_input + 'turnover.h5', key='df')
IDs = list(turnover['ID'])
dates = list(turnover.columns)[2:]

last_dates = []   # last dates of each month
first_dates = []  # first dates of each month
date_range = {}   # at last dates of each month, look back 20 days

for i in range(1, len(dates) - 1):
    if str(dates[i])[:6] != str(dates[i + 1])[:6]:
        last_dates.append(dates[i])
        first_dates.append(dates[i + 1])
        if i > 20:
            date_range[dates[i]] = []
            for j in range(i - 19, i + 1):
                date_range[dates[i]].append(dates[j])

STR = pd.DataFrame(columns=last_dates[1:], index=IDs)
turnover = turnover.replace(0.0, np.nan)

for i in range(len(IDs)):
    turn = turnover.iloc[i, :]
    
    for j in range(1, len(last_dates)):
        turnover_data = turn.loc[date_range[last_dates[j]]]
        num = turnover_data.count()
        if num < 10:
            STR.loc[IDs[i], last_dates[j]] = np.nan
        else:
            STR.loc[IDs[i], last_dates[j]] = turnover_data.std()
STR.to_hdf(dir_data_output + 'STR.h5', key='df')

# ========================
# Calculate monthly return
dir_data_input = dir_main + 'result/data/'
dir_data_output = dir_main + 'result/'

close = pd.read_hdf(dir_data_input + 'close_after.h5', key='df')
open_ = pd.read_hdf(dir_data_input + 'open_after.h5', key='df')
IDs = list(close['ID'])
dates = list(close.columns)[2:]

last_dates = []   # last dates of each month
first_dates = []  # first dates of each month
date_range = {}   # at last dates of each month, look back 20 days

for i in range(1, len(dates) - 1):
    if str(dates[i])[:6] != str(dates[i + 1])[:6]:
        last_dates.append(dates[i])
        first_dates.append(dates[i + 1])
        if i > 20:
            date_range[dates[i]] = []
            for j in range(i - 19, i + 1):
                date_range[dates[i]].append(dates[j])

ret = pd.DataFrame(columns=last_dates[1:], index=IDs)

for i in range(len(IDs)):
    price0 = open_.iloc[i, :]
    price1 = close.iloc[i, :]
    
    for j in range(1, len(last_dates) - 1):
        begin_price = price0[first_dates[j]]
        end_price = price1[last_dates[j + 1]]
        ret.loc[IDs[i], last_dates[j]] = (end_price - begin_price) / begin_price
    
    begin_price = price0[first_dates[-1]]
    end_price = price1[20210630]
    ret.loc[IDs[i], last_dates[-1]] = (end_price - begin_price) / begin_price
ret.to_hdf(dir_data_output + 'ret.h5', key='df')

# ===================
# STR size neutrality
dir_data_input = dir_main + 'result/data/'
dir_data_output = dir_main + 'result/'

model = linear_model.LinearRegression()
STR = pd.read_hdf(dir_data_output + 'STR.h5', key='df')
STR.reset_index(inplace = True, drop = True)
capital = pd.read_hdf(dir_data_input + 'capital.h5', key='df')
close = pd.read_hdf(dir_data_input + 'close_before.h5', key='df')
STR_deSize = pd.DataFrame(columns=last_dates[1:], index=range(len(IDs)))

for i in range(1, len(last_dates)):
    y = STR[last_dates[i]]
    cap = capital[last_dates[i]]
    price = close[last_dates[i]]
    size = cap * price
    x = np.log(size.replace(0.0, np.nan))
    temp = x.isna() | y.isna()
    x = x[temp == False].values.reshape(-1, 1)
    y = y[temp == False]
    model.fit(x, y)
    y_predict = model.predict(x)
    STR_deSize[last_dates[i]][temp == False] = y - y_predict

STR_deSize = STR_deSize.rename(index = dict(zip(range(len(IDs)), IDs)))
STR_deSize.to_hdf(dir_data_output + 'STR_deSize.h5', key='df')

# ========================================
# A simpler way to calculate factor return
dir_data_input = dir_main + 'result/'
dir_data_output = dir_main + 'result/'
ret = pd.read_hdf(dir_data_input + 'ret.h5', key='df')

def calc_ret_simple(factor, num):
    data = pd.read_hdf(dir_data_input + factor + '.h5', key='df')
    all_dates = list(data.columns)
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

calc_ret_simple('STR_deSize', 10)

# =====================
# Calculate IC / RankIC
dir_data_input = dir_main + 'result/'
dir_data_output = dir_main + 'result/'
ret = pd.read_hdf(dir_data_input + 'ret.h5', key='df')

def calc_IC(factor):
    data = pd.read_hdf(dir_data_output + factor + '.h5', key='df')
    ICs = []
    
    # for date in last_dates[1:]:
    for date in last_dates[24 : -1]:
        x = data[date].astype('float64')
        y = ret[date].astype('float64')
        ICs.append(x.corr(y))
    return np.mean(ICs)

def calc_RankIC(factor):
    data = pd.read_hdf(dir_data_output + factor + '.h5', key='df')
    RankICs = []
    
    # for date in last_dates[1:]:
    for date in last_dates[24 : -1]:
        x = data[date].rank()
        y = ret[date].rank()
        RankICs.append(x.corr(y))
    return np.mean(RankICs)

print(calc_IC('STR_deSize'))
print(calc_RankIC('STR_deSize'))

# =============================
# Calculate information entropy
dir_data_input = dir_main + 'result/data/'
dir_data_output = dir_main + 'result/'

turnover = pd.read_hdf(dir_data_input + 'turnover.h5', key='df')
turnover = turnover.replace(0.0, np.nan)
IDs = list(turnover['ID'])
dates = list(turnover.columns)[2:]

last_dates = []   # last dates of each month
first_dates = []  # first dates of each month
date_range = {}   # at last dates of each month, look back 20 days

for i in range(1, len(dates) - 1):
    if str(dates[i])[:6] != str(dates[i + 1])[:6]:
        last_dates.append(dates[i])
        first_dates.append(dates[i + 1])
        if i > 20:
            date_range[dates[i]] = []
            for j in range(i - 19, i + 1):
                date_range[dates[i]].append(dates[j])

def calc_entropy(min_data_num, split_num):
    entropy = pd.DataFrame(columns=last_dates[1:], index=IDs)
    for i in range(len(IDs)):
        turn = turnover.iloc[i, :]
        
        for j in range(1, len(last_dates)):
            turnover_data = turn.loc[date_range[last_dates[j]]]
            num = turnover_data.count()
            
            if num < min_data_num:
                entropy.loc[IDs[i], last_dates[j]] = np.nan
            else:
                turnover_data = turnover_data[turnover_data.notnull()]
                hist = np.histogram(turnover_data, split_num)[0] / num
                H = 0.0
                for k in hist:
                    if k != 0.0:
                        H = H - k * np.log2(k)
                H = H / np.log2(split_num)
                entropy.loc[IDs[i], last_dates[j]] = H
    return entropy
