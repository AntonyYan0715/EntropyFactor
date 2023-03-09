import numpy as np
import pandas as pd
from math import ceil

dir_main = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/'
dir_data_input = dir_main + 'data/'
dir_data_output = dir_main + 'result/'

close = pd.read_hdf(dir_data_input + 'close_process.h5', key='df')
open_ = pd.read_hdf(dir_data_input + 'open_process.h5', key='df')
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

# Portfolio
close.set_index(['ID'], inplace=True)
open_.set_index(['ID'], inplace=True)
ret = pd.read_hdf(dir_data_output + 'ret.h5', key='df')

def calc_ret(factor):
    data = pd.read_hdf(dir_data_output + factor + '.h5', key='df')
    monthly_ret = [[], [], [], [], []]
    long = []
    two = []
    three = []
    four = []
    short = []

    for i in range(1, len(last_dates)):
        date = last_dates[i]
        stock_num = data[date].count()
        percent = ceil(stock_num * 0.2)
        stock_pool = list(data[date].sort_values().index)
    
        Long = stock_pool[0 : percent]
        Two = stock_pool[percent : 2*percent]
        Three = stock_pool[2*percent : 3*percent]
        Four = stock_pool[3*percent : 4*percent]
        Short = stock_pool[4*percent : stock_num]

        for each in Long[:]:
            Open = open_.loc[each][first_dates[i - 1]]
            Close = close.loc[each][last_dates[i - 1]]
            if (Open - Close) / Close > 0.0985:
                Long.remove(each)
        
        for each in Two[:]:
            Open = open_.loc[each][first_dates[i - 1]]
            Close = close.loc[each][last_dates[i - 1]]
            if (Open - Close) / Close > 0.0985:
                Two.remove(each)

        for each in Three[:]:
            Open = open_.loc[each][first_dates[i - 1]]
            Close = close.loc[each][last_dates[i - 1]]
            if (Open - Close) / Close > 0.0985:
                Three.remove(each)
            
        for each in Four[:]:
            Open = open_.loc[each][first_dates[i - 1]]
            Close = close.loc[each][last_dates[i - 1]]
            if (Open - Close) / Close > 0.0985:
                Four.remove(each)
        
        for each in Short[:]:
            Open = open_.loc[each][first_dates[i - 1]]
            Close = close.loc[each][last_dates[i - 1]]
            if (Open - Close) / Close > 0.0985:
                Short.remove(each)
        
        Long = Long + long
        Two = Two + two
        Three = Three + three
        Four = Four + four
        Short = Short + short
            
        rtn = ret[date]
        monthly_ret[0].append(rtn[Long].mean())
        monthly_ret[1].append(rtn[Two].mean())
        monthly_ret[2].append(rtn[Three].mean())
        monthly_ret[3].append(rtn[Four].mean())
        monthly_ret[4].append(rtn[Short].mean())
        
        long = []
        two = []
        three = []
        four = []
        short = []
        
        for each in stock_pool[0 : percent]:
            Open = open_.loc[each][last_dates[i]]
            Close = close.loc[each][last_dates[i]]
            if (Close - Open) / Open < -0.0985:
                long.append(each)
        
        for each in stock_pool[percent : 2*percent]:
            Open = open_.loc[each][last_dates[i]]
            Close = close.loc[each][last_dates[i]]
            if (Close - Open) / Open < -0.0985:
                two.append(each)
                
        for each in stock_pool[2*percent : 3*percent]:
            Open = open_.loc[each][last_dates[i]]
            Close = close.loc[each][last_dates[i]]
            if (Close - Open) / Open < -0.0985:
                three.append(each)
                
        for each in stock_pool[3*percent : 4*percent]:
            Open = open_.loc[each][last_dates[i]]
            Close = close.loc[each][last_dates[i]]
            if (Close - Open) / Open < -0.0985:
                four.append(each)
        
        for each in stock_pool[4*percent : stock_num]:
            Open = open_.loc[each][last_dates[i]]
            Close = close.loc[each][last_dates[i]]
            if (Close - Open) / Open < -0.0985:
                short.append(each)

    all_dates = last_dates[2:]
    all_dates.append(dates[-1])
    ret_df = pd.DataFrame({'date':all_dates, 'Long':monthly_ret[0], '2':monthly_ret[1],\
                           '3':monthly_ret[2], '4':monthly_ret[3], 'Short':monthly_ret[4],\
                           'Long-Short':list(map(lambda x: x[0]-x[1], zip(monthly_ret[0], monthly_ret[4])))})
    ret_df.to_excel(dir_data_output + factor + '.xlsx', index=False)

# A simpler way to calculate return
def calc_ret_simple(factor):
    data = pd.read_hdf(dir_data_output + factor + '.h5', key='df')
    monthly_ret = [[], [], [], [], []]
    
    for i in range(1, len(last_dates)):
        date = last_dates[i]
        stock_num = data[date].count()
        percent = ceil(stock_num * 0.2)
        stock_pool = list(data[date].sort_values().index)
        
        Long = stock_pool[0 : percent]
        Two = stock_pool[percent : 2*percent]
        Three = stock_pool[2*percent : 3*percent]
        Four = stock_pool[3*percent : 4*percent]
        Short = stock_pool[4*percent : stock_num]
        
        rtn = ret[date]
        monthly_ret[0].append(rtn[Long].mean())
        monthly_ret[1].append(rtn[Two].mean())
        monthly_ret[2].append(rtn[Three].mean())
        monthly_ret[3].append(rtn[Four].mean())
        monthly_ret[4].append(rtn[Short].mean())
        
    all_dates = last_dates[2:]
    all_dates.append(dates[-1])
    ret_df = pd.DataFrame({'date':all_dates, 'Long':monthly_ret[0], '2':monthly_ret[1],\
                           '3':monthly_ret[2], '4':monthly_ret[3], 'Short':monthly_ret[4],\
                           'Long-Short':list(map(lambda x: x[0]-x[1], zip(monthly_ret[0], monthly_ret[4])))})
    ret_df.to_excel(dir_data_output + factor + '.xlsx', index=False)

calc_ret_simple('Turn20')
calc_ret_simple('Turn20_deSize')
calc_ret_simple('STR_deSize')

# Calculate return in 10 groups
def calc_ret_10(factor):
    data = pd.read_hdf(dir_data_output + factor + '.h5', key='df')
    monthly_ret = [[], [], [], [], [], [], [], [], [], []]
    
    for i in range(1, len(last_dates)):
        date = last_dates[i]
        stock_num = data[date].count()
        percent = int(stock_num * 0.1)
        stock_pool = list(data[date].sort_values().index)
        
        Long = stock_pool[0 : percent]
        Two = stock_pool[percent : 2*percent]
        Three = stock_pool[2*percent : 3*percent]
        Four = stock_pool[3*percent : 4*percent]
        Five = stock_pool[4*percent : 5*percent]
        Six = stock_pool[5*percent : 6*percent]
        Seven = stock_pool[6*percent : 7*percent]
        Eight = stock_pool[7*percent : 8*percent]
        Nine = stock_pool[8*percent : 9*percent]
        Short = stock_pool[9*percent : stock_num]
        
        rtn = ret[date]
        monthly_ret[0].append(rtn[Long].mean())
        monthly_ret[1].append(rtn[Two].mean())
        monthly_ret[2].append(rtn[Three].mean())
        monthly_ret[3].append(rtn[Four].mean())
        monthly_ret[4].append(rtn[Five].mean())
        monthly_ret[5].append(rtn[Six].mean())
        monthly_ret[6].append(rtn[Seven].mean())
        monthly_ret[7].append(rtn[Eight].mean())
        monthly_ret[8].append(rtn[Nine].mean())
        monthly_ret[9].append(rtn[Short].mean())
        
    all_dates = last_dates[2:]
    all_dates.append(dates[-1])
    ret_df = pd.DataFrame({'date':all_dates, 'Long':monthly_ret[0], '2':monthly_ret[1],\
                           '3':monthly_ret[2], '4':monthly_ret[3], '5':monthly_ret[4],\
                           '6':monthly_ret[5], '7':monthly_ret[6], '8':monthly_ret[7],\
                           '9':monthly_ret[8], 'Short':monthly_ret[9],\
                           'Long-Short':list(map(lambda x: x[0]-x[1], zip(monthly_ret[0], monthly_ret[9])))})
    ret_df.to_excel(dir_data_output + factor + '.xlsx', index=False)

calc_ret_10('STR_deSize')