import numpy as np
import pandas as pd
from copy import deepcopy

dir_main = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/'

# Turnover rate data
df1 = pd.read_hdf(dir_main + 'turnover.h5', 'df')

ID = list(df1['ID'])
dates = list(df1.columns[2:])
last_dates = []  # last dates of each month
date_range = {}

for i in range(1, len(dates) - 1):
    if str(dates[i])[:6] != str(dates[i + 1])[:6]:
        last_dates.append(dates[i])
        date_range[dates[i]] = range(i - 20, i)

# Close price data
df2 = pd.read_hdf(dir_main + 'close_after.h5', 'df')

# ===================
# Test ID = 000001.SZ
turnover = df1.iloc[1, :]
close = df2.iloc[1, :]
STR = {}
entropy = {}
return_monthly = {}

# Calculate monthly STR(Std of Turnover Rate)
for i in range(1, len(last_dates)):
    turnover_20 = []
    for j in date_range[last_dates[i]]:
        turnover_20.append(turnover[dates[j]])
    STR[last_dates[i]] = np.std(turnover_20)

# Calculate monthly information entropy
n = 5
for i in range(1, len(last_dates)):
    turnover_20 = []
    for j in date_range[last_dates[i]]:
        turnover_20.append(turnover[dates[j]])
    if len(turnover_20) == 20:
        hist = np.histogram(turnover_20, n)[0] / 20
        H = 0.0
        for k in hist:
            if k != 0.0:
                H = H - k * np.log2(k)
        H = H / np.log2(n)
        entropy[last_dates[i]] = H

# Calculate monthly return
for i in range(1, len(last_dates) - 1):
    date = last_dates[i]
    next_date = last_dates[i + 1]
    price = close[date]
    next_price = close[next_date]
    return_monthly[date] = next_price / price - 1

return_monthly[last_dates[-1]] = close[dates[-1]] / close[last_dates[-1]] - 1

cor1 = np.corrcoef(list(STR.values()), list(return_monthly.values()))[0,1]
cor2 = np.corrcoef(list(entropy.values()), list(return_monthly.values()))[0,1]


# =================
# Calculate mean IC
IDs = list(df1['ID'])
STR_df = pd.DataFrame(columns = last_dates[1:], index = IDs)
entropy_df = pd.DataFrame(columns = last_dates[1:], index = IDs)
ret_df = pd.DataFrame(columns = last_dates[1:], index = IDs)

for i in range(len(IDs)):
    turnover = df1.iloc[i, :]
    close = df2.iloc[i, :]
    
    for j in range(1, len(last_dates)):
        turnover_20 = []
        
        for k in date_range[last_dates[j]]:
            turnover_20.append(turnover[dates[k]])
        
        STR_df.loc[IDs[i], last_dates[j]] = np.std(turnover_20)
        hist = np.histogram(turnover_20, 5)[0] / 20
        H = 0.0
        for l in hist:
            if l != 0.0:
                H = H - l * np.log2(l)
        H = H / np.log2(5)
        entropy_df.loc[IDs[i], last_dates[j]] = H

    for j in range(1, len(last_dates) - 1):
        date = last_dates[j]
        next_date = last_dates[j + 1]
        price = close[date]
        next_price = close[next_date]
        ret_df.loc[IDs[i], date] = next_price / price - 1
    ret_df.loc[IDs[i], last_dates[-1]] = close[dates[-1]] / close[last_dates[-1]] - 1


# ==========================================
# Calculate factors' correlation coefficient
dir_data_input = dir_main + 'result/'
Turn20_deSize = pd.read_hdf(dir_data_input + 'Turn20_deSize.h5')
Turn20_standard = pd.read_hdf(dir_data_input + 'Turn20_standard.h5')
dates = list(Turn20_standard.index)
Turn20_deSize = Turn20_deSize.astype('float64')

ICs = []
for date in dates:
    x = Turn20_standard.loc[date]
    y = Turn20_deSize[date]
    ICs.append(x.corr(y))
print(np.mean(ICs))


dir_data_input = dir_main + 'result/'
STR_deSize = pd.read_hdf(dir_data_input + 'STR_deSize.h5')
STR_standard = pd.read_hdf(dir_data_input + 'STR_standard.h5')
dates = list(STR_standard.index)
STR_deSize = STR_deSize.astype('float64')

ICs = []
for date in dates:
    x = STR_standard.loc[date]
    y = STR_deSize[date]
    ICs.append(x.corr(y))
print(np.mean(ICs))

RankICs = []
for date in dates:
    x = STR_standard.loc[date].rank()
    y = STR_deSize[date].rank()
    RankICs.append(x.corr(y))
print(np.mean(RankICs))