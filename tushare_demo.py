import tushare as ts

# 000300.XSHG

print(ts.__version__)

ts.set_token('726e1ea55025ba6def26cdcaba3e8531b29e6c50e6ea36e0f034b2e9')

pro = ts.pro_api()

df = pro.daily(ts_code='002264.SZ', start_date='20210101', end_date='20211207')

# #多个股票
# df = pro.daily(ts_code='000001.SZ,600000.SH', start_date='20180701', end_date='20180718')
# # 或者

# df = pro.query('daily', ts_code='000001.SZ', start_date='20180701', end_date='20180718')
# # 也可以通过日期取历史某一天的全部历史

# df = pro.daily(trade_date='20180810')

# print(df)
df.dropna()
df.to_csv('C:/Users/DK/Desktop/LSTM/lstm/sh300.csv')

print(df.columns)

print(df.describe())

# df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')

# df = pro.query('trade_cal', exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')

