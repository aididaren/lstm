
# https://blog.csdn.net/qq_43344047/article/details/118343959?spm=1001.2101.3001.6650.18&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-18.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-18.no_search_link
# https://blog.csdn.net/qq_43344047/article/details/118607710?spm=1001.2014.3001.5501
import tushare as ts
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

ts.set_token('208cf0f7e03acf9024568071ab959d8cf3d1908385a7009906dc66d5')
pro = ts.pro_api()

time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
end_dt = time_temp.strftime('%Y%m%d')

# 准备训练集数据

df = ts.pro_bar(ts_code='002264.SZ', start_date='20210101',
                end_date='20211206', freq='D')
df.head()  # 用 df.head() 可以查看一下下载下来的股票价格数据，显示数据如下：

# 把数据按时间调转顺序，最新的放后面
df = df.iloc[::-1]
df.reset_index(inplace=True)
# print(df)

training_set = df.loc[:, ['close']]
# 只取价格数据，不要表头等内容
training_set = training_set.values
# 数据归一化 https://blog.csdn.net/weixin_40683253/article/details/81508321
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# print(training_set_scaled)

X_train = []
y_train = []
devia = 10
for i in range(devia, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-devia:i])
    y_train.append(training_set_scaled[i, training_set_scaled.shape[1] - 1])

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
print(y_train.shape)


class LSTMNET(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=160, output_size=1, num_layers=8):
        super(LSTMNET, self).__init__()
        self.lstm_layer = nn.LSTM(
            input_size, hidden_layer_size, num_layers, batch_first=True)
        self.out_layer1 = nn.Linear(hidden_layer_size, output_size, bias=True)
        self.out_layer2 = nn.Linear(num_layers, output_size, bias=True)

    def forward(self, share):
        out, (h, c) = self.lstm_layer(share.to(torch.float32))
        out = h
        a, b, c = out.shape
        out = out.reshape(b, a)
        out = self.out_layer2(out)
        return out


for i in range(epochs):
    for shares, labels in train_loader:
        train = Variable(shares, requires_grad=True)
        labels = Variable(labels, requires_grad=True)
        outputs = model(train)

        loss_mse = loss_MSE(outputs.float(), labels.float())
        loss_rmse = torch.sqrt(loss_mse)
        loss_mae = loss_MAE(outputs.float(), labels.float())

        optimizer.zero_grad()
        loss_rmse.backward()
        optimizer.step()

        count = count+1
        if count % 100 == 0:
            rmse.append(loss_rmse.data)
            mae.append(loss_mae.data)

        if i == epochs*bs-1:
            a, _ = outputs.shape
            for j in range(a):
                pred.append(outputs[j])
                actl.append(labels[j])

    print("epoch:{}    Train:  RMSE:{:.8f}  MAE:{:.8f} ".format(
        i, loss_rmse.data, loss_mae.data))


plt.figure()
plt.plot(pred, "r-")
plt.plot(actl, "b-")

plt.figure()
plt.plot(rmse, "r-")
plt.plot(mae, "b--")
