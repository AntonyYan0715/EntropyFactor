import numpy as np
import pandas as pd

dir_main = 'C:/Users/11765/Desktop/SoochowSecurities/entropy/'
dir_data_input = dir_main + 'data/'
dir_data_output = dir_main + 'result/'

close = pd.read_hdf(dir_data_input + 'close_after.h5', key='df')
dates = list(close.columns)[2:]

# 剔除未满60个交易日的次新股
df1 = pd.read_hdf(dir_data_input + 'trading_day.h5', key='df')
close[df1[dates] < 60] = np.nan

# 剔除ST股
df2 = pd.read_hdf(dir_data_input + 'ST.h5', key='df')
close[df2[dates] == 1] = np.nan

# 剔除非正常交易股票
df3 = pd.read_hdf(dir_data_input + 'trading_state.h5', key='df')
close[df3[dates] != '交易'] = np.nan

close.to_hdf(dir_data_output + 'close_process.h5', key='df')

# 处理Turn20因子
Turn20 = pd.read_hdf(dir_data_input + 'Turn20.h5', key='df')
dates = list(Turn20.columns)
IDs = list(Turn20.index)
Turn20 = Turn20.rename(index=dict(zip(IDs, range(len(IDs)))))

df1 = pd.read_hdf(dir_data_input + 'trading_day.h5', key='df')
Turn20[df1[dates] < 60] = np.nan

df2 = pd.read_hdf(dir_data_input + 'ST.h5', key='df')
Turn20[df2[dates] == 1] = np.nan

df3 = pd.read_hdf(dir_data_input + 'trading_state.h5', key='df')
Turn20[df3[dates] != '交易'] = np.nan

Turn20 = Turn20.rename(index=dict(zip(range(len(IDs)), IDs)))
Turn20.to_hdf(dir_data_output + 'Turn20.h5', key='df')

# 处理STR因子
STR = pd.read_hdf(dir_data_input + 'STR.h5', key='df')
dates = list(STR.columns)
IDs = list(STR.index)
STR = STR.rename(index=dict(zip(IDs, range(len(IDs)))))

df1 = pd.read_hdf(dir_data_input + 'trading_day.h5', key='df')
STR[df1[dates] < 60] = np.nan

df2 = pd.read_hdf(dir_data_input + 'ST.h5', key='df')
STR[df2[dates] == 1] = np.nan

df3 = pd.read_hdf(dir_data_input + 'trading_state.h5', key='df')
STR[df3[dates] != '交易'] = np.nan

STR = STR.rename(index=dict(zip(range(len(IDs)), IDs)))
STR.to_hdf(dir_data_output + 'STR.h5', key='df')