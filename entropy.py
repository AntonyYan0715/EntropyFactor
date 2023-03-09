import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

dir_main = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/'
dir_data_output = dir_main + 'result/'

# =============================
# Calculate information entropy
dir_data_input = dir_main + 'result/data/'

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
        print(i)
        turn = turnover.iloc[i, :]
        
        for j in range(1, len(last_dates)):
            turnover_data = turn.loc[date_range[last_dates[j]]]
            num = turnover_data.count()
            
            if num < min_data_num:
                entropy.loc[IDs[i], last_dates[j]] = np.nan
            else:
                turnover_data = turnover_data[turnover_data.notnull()]
                hist = np.histogram(turnover_data, split_num)[0] / num
                H = -np.nansum(hist * np.log2(hist)) / np.log2(split_num)
                entropy.loc[IDs[i], last_dates[j]] = H
    return entropy

H1 = calc_entropy(15, 5)
H2 = calc_entropy(18, 5)
H3 = calc_entropy(20, 5)

# =====================
# Calculate IC / RankIC
dir_data_input = dir_main + 'result/'
ret = pd.read_hdf(dir_data_input + 'ret.h5', key='df')

def calc_IC(factor):
    ICs = []
    
    # for date in last_dates[1:]:
    for date in last_dates[24 : -1]:
        x = factor[date].astype('float64')
        y = ret[date].astype('float64')
        ICs.append(x.corr(y))
    return np.mean(ICs)

def calc_RankIC(factor):
    RankICs = []
    
    # for date in last_dates[1:]:
    for date in last_dates[24 : -1]:
        x = factor[date].rank()
        y = ret[date].rank()
        RankICs.append(x.corr(y))
    return np.mean(RankICs)

print(calc_IC(H1))
print(calc_RankIC(H1))

# ==========================================
# Calculate factors' correlation coefficient
dir_data_input = dir_main + 'result/'
STR = pd.read_hdf(dir_data_input + 'STR.h5', key='df')

dates = list(H1.columns)
cor_coef = []
for date in dates:
    x = H1[date].astype('float64')
    y = STR[date].astype('float64')
    cor_coef.append(x.corr(y))
print(np.mean(cor_coef))

# =======================
# Picture of distribution
n, bins, patches = plt.hist(x=values, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
maxfreq = n.max()

# limit of y-axis
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# =======================
# Entropy size neutrality
dir_data_input = dir_main + 'result/data/'
dir_data_output = dir_main + 'result/'

model = linear_model.LinearRegression()
H1.reset_index(inplace = True, drop = True)
capital = pd.read_hdf(dir_data_input + 'capital.h5', key='df')
close = pd.read_hdf(dir_data_input + 'close_before.h5', key='df')
H1_deSize = pd.DataFrame(columns=last_dates[1:], index=range(len(IDs)))

for i in range(1, len(last_dates)):
    y = H1[last_dates[i]]
    cap = capital[last_dates[i]]
    price = close[last_dates[i]]
    size = cap * price
    x = np.log(size.replace(0.0, np.nan))
    temp = x.isna() | y.isna()
    x = x[temp == False].values.reshape(-1, 1)
    y = y[temp == False]
    model.fit(x, y)
    y_predict = model.predict(x)
    H1_deSize[last_dates[i]][temp == False] = y - y_predict

H1_deSize = H1_deSize.rename(index = dict(zip(range(len(IDs)), IDs)))
H1_deSize.to_hdf(dir_data_output + 'entropy_deSize.h5', key='df')

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

calc_ret_simple('entropy_deSize', 10)

# ==============================
# Entropy (Max - Min) neutrality
dir_data_input = dir_main + 'result/data/'
dir_data_output = dir_main + 'result/'

turnover = pd.read_hdf(dir_data_input + 'turnover.h5', key='df')
turnover = turnover.replace(0.0, np.nan)

def calc_Max_Min(min_data_num):
    Max_Min = pd.DataFrame(columns=last_dates[1:], index=IDs)
    for i in range(len(IDs)):
        print(i)
        turn = turnover.iloc[i, :]
        
        for j in range(1, len(last_dates)):
            turnover_data = turn.loc[date_range[last_dates[j]]]
            num = turnover_data.count()
            
            if num < min_data_num:
                Max_Min.loc[IDs[i], last_dates[j]] = np.nan
            else:
                Max_Min.loc[IDs[i], last_dates[j]] = turnover_data.max() - turnover_data.min()
    return Max_Min

Max_Min1 = calc_Max_Min(18)
model = linear_model.LinearRegression()
H_deMax = pd.DataFrame(columns=last_dates[1:], index=IDs)

for i in range(1, len(last_dates)):
    y = H2[last_dates[i]]
    x = Max_Min1[last_dates[i]]
    temp = x.isna() | y.isna()
    x = x[temp == False].values.reshape(-1, 1)
    y = y[temp == False]
    model.fit(x, y)
    y_predict = model.predict(x)
    H_deMax[last_dates[i]][temp == False] = y - y_predict

H_deMax.to_hdf(dir_data_output + 'entropy_deMax.h5', key='df')

# =============================================
# A new method to calculate information entropy
turnover = turnover.rename(index = dict(zip(range(len(IDs)), IDs)))
del turnover['ID'], turnover['日期']

def calc_entropy_new(min_data_num, split_num):
    entropy = pd.DataFrame(columns=last_dates[1:], index=IDs)
    for i in range(1, len(last_dates)):
        print(i)
        data = turnover[date_range[last_dates[i]]]
        Max = data.max().max()
        Min = data.min().min()
        
        for ID in IDs:
            turnover_data = data.loc[ID, :]
            num = turnover_data.count()
            
            if num < min_data_num:
                entropy.loc[ID, last_dates[i]] = np.nan
            else:
                turnover_data = turnover_data[turnover_data.notnull()]
                hist = np.histogram(turnover_data, bins=split_num, range=(Min, Max))[0] / num
                H = -np.nansum(hist * np.log2(hist)) / np.log2(split_num)
                entropy.loc[ID, last_dates[i]] = H
    return entropy

H1 = calc_entropy_new(20, 10)  # RankIC = -0.0807
H2 = calc_entropy_new(18, 8)   # RankIC = -0.0798
H3 = calc_entropy_new(18, 10)  # RankIC = -0.0802, ret = 285.75
H4 = calc_entropy_new(18, 12)  # RankIC = -0.0795
H5 = calc_entropy_new(18, 15)  # RankIC = -0.0788
H6 = calc_entropy_new(18, 20)  # RankIC = -0.0779, ret = 165.59
H7 = calc_entropy_new(18, 30)  # RankIC = -0.0759, ret = 153.48
H8 = calc_entropy_new(15, 10)  # RankIC = -0.0799, ret = 275.79
H9 = calc_entropy_new(15, 20)  # RankIC = -0.0778, ret = 156.26

# =======================================
# optimize parameters [look back 60 days]
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
        if i > 60:
            date_range[dates[i]] = []
            for j in range(i - 59, i + 1):
                date_range[dates[i]].append(dates[j])

turnover = turnover.rename(index = dict(zip(range(len(IDs)), IDs)))
del turnover['ID'], turnover['日期']

def calc_entropy_optimize(min_data_num, split_num):
    entropy = pd.DataFrame(columns=last_dates[3:], index=IDs)
    for i in range(3, len(last_dates)):
        print(i)
        data = turnover[date_range[last_dates[i]]]
        Max = data.max().max()
        Min = data.min().min()
        
        for ID in IDs:
            turnover_data = data.loc[ID, :]
            num = turnover_data.count()
            
            if num < min_data_num:
                entropy.loc[ID, last_dates[i]] = np.nan
            else:
                turnover_data = turnover_data[turnover_data.notnull()]
                hist = np.histogram(turnover_data, bins=split_num, range=(Min, Max))[0] / num
                H = -np.nansum(hist * np.log2(hist)) / np.log2(split_num)
                entropy.loc[ID, last_dates[i]] = H
    return entropy

H1 = calc_entropy_optimize(55, 20)   # RankIC = -0.0599
H2 = calc_entropy_optimize(55, 30)   # RankIC = -0.0585
H3 = calc_entropy_optimize(50, 20)   # RankIC = -0.0594
H4 = calc_entropy_optimize(50, 30)   # RankIC = -0.0582
H5 = calc_entropy_optimize(50, 50)   # RankIC = -0.0574
H6 = calc_entropy_optimize(50, 100)  # RankIC = -0.0569
H7 = calc_entropy_optimize(50, 120)  # RankIC = -0.0566

# =======================================
# optimize parameters [look back 40 days]
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
        if i > 40:
            date_range[dates[i]] = []
            for j in range(i - 39, i + 1):
                date_range[dates[i]].append(dates[j])

turnover = turnover.rename(index = dict(zip(range(len(IDs)), IDs)))
del turnover['ID'], turnover['日期']

def calc_entropy_optimize(min_data_num, split_num):
    entropy = pd.DataFrame(columns=last_dates[2:], index=IDs)
    for i in range(2, len(last_dates)):
        print(i)
        data = turnover[date_range[last_dates[i]]]
        Max = data.max().max()
        Min = data.min().min()
        
        for ID in IDs:
            turnover_data = data.loc[ID, :]
            num = turnover_data.count()
            
            if num < min_data_num:
                entropy.loc[ID, last_dates[i]] = np.nan
            else:
                turnover_data = turnover_data[turnover_data.notnull()]
                hist = np.histogram(turnover_data, bins=split_num, range=(Min, Max))[0] / num
                H = -np.nansum(hist * np.log2(hist)) / np.log2(split_num)
                entropy.loc[ID, last_dates[i]] = H
    return entropy

H1 = calc_entropy_optimize(35, 20)   # RankIC = -0.0672
H2 = calc_entropy_optimize(35, 30)   # RankIC = -0.0659
H3 = calc_entropy_optimize(35, 50)   # RankIC = -0.0651
H4 = calc_entropy_optimize(35, 80)   # RankIC = -0.0646
H5 = calc_entropy_optimize(30, 20)   # RankIC = -0.0669
H6 = calc_entropy_optimize(30, 50)   # RankIC = -0.0648
