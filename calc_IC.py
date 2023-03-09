import numpy as np
import pandas as pd
from math import ceil

dir_main = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/'
dir_data_input = dir_main + 'data/'
dir_data_output = dir_main + 'result/'

close = pd.read_hdf(dir_data_input + 'close_process.h5', key='df')
IDs = list(close['ID'])
dates = list(close.columns)[2:]

last_dates = []   # last dates of each month
first_dates = []  # first dates of each month
date_range = {}

for i in range(1, len(dates) - 1):
    if str(dates[i])[:6] != str(dates[i + 1])[:6]:
        last_dates.append(dates[i])
        first_dates.append(dates[i + 1])
        if i > 20:
            date_range[dates[i]] = []    
            for j in range(i - 19, i + 1):
                date_range[dates[i]].append(dates[j])

# factor
ret = pd.read_hdf(dir_data_output + 'ret.h5', key='df')

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
    

print(calc_IC('Ret20'))
print(calc_IC('Turn20'))
print(calc_IC('Turn20_deSize'))
print(calc_IC('STR_deSize'))

print(calc_RankIC('Ret20'))
print(calc_RankIC('Turn20'))
print(calc_RankIC('Turn20_deSize'))
print(calc_RankIC('STR_deSize'))