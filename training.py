# %%
# Khai báo thư viện
import pandas as pd  # Đọc dữ liệu
import numpy as np  # Xử lý dữ liệu
import datetime as dt  # Xử lý thời gian
import matplotlib.pyplot as plt  # Vẽ biểu đồ
import pandas_datareader as pdr  # Lấy dữ liệu từ api
from sklearn.preprocessing import MinMaxScaler  # Chuẩn hóa dữ liệu
from keras.callbacks import ModelCheckpoint  # Lưu lại huấn luyện tốt nhất
from keras.models import load_model  # tải mô hình

# Khai báo các lớp để xây dựng mô hình học sâu
from keras.models import Sequential  # Đầu vào
from keras.layers import LSTM  # Mô hình học phục thuộc có giám sát
from keras.layers import Dropout  # tránh học tủ
from keras.layers import Dense  # đầu ra

# công cụ kiểm tra độ chính xác mô hình
from sklearn.metrics import r2_score  # đo mức độ phù hợp
from sklearn.metrics import mean_absolute_error  # đo sai số tuyệt đối trung bình
# phần trăm sai số tuyệt đối trung bình
from sklearn.metrics import mean_absolute_percentage_error
# %%
# dữ liệu được lấy từ api của tiingo
start = '01/01/2015'
end = dt.datetime.now().strftime("%d/%m/%Y")
company = 'AAPL'
key = '3848aa6f3355ce8a788bdd508d862b26358c1963'
df = pdr.get_data_tiingo(company, start=start, end=end, api_key=key)
df.to_csv("data.csv")
# %%
# đọc dữ liệu từ file *.csv
df = pd.read_csv("data.csv")
df["date"] = pd.to_datetime(df.date, format="%Y-%m-%d %H:%M:%S")
df["date"] = df["date"].dt.strftime('%Y-%m-%d')


def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return dt.datetime(year=year, month=month, day=day)


df['date'] = df['date'].apply(str_to_datetime)
dates = df['date']
df.set_index('date', drop=True, inplace=True)
df.head()
# %%
df.describe()
# %%
# Preprocess data
df = df[['close', 'high', 'low', 'open']]
df = pd.DataFrame(df)
df['H-L'] = df['high'] - df['low']
df['O-C'] = df['open'] - df['close']
# Moving average in 7, 14, 21 day
ma_1 = 7
ma_2 = 14
ma_3 = 21
df[f'SMA_{ma_1}'] = df['close'].rolling(window=ma_1).mean()
df[f'SMA_{ma_2}'] = df['close'].rolling(window=ma_2).mean()
df[f'SMA_{ma_3}'] = df['close'].rolling(window=ma_3).mean()
# Standard devigation
df[f'SD_{ma_1}'] = df['close'].rolling(window=ma_1).std()
df[f'SD_{ma_3}'] = df['close'].rolling(window=ma_3).std()
# Drop data lack of value
df.dropna(inplace=True)
# print("Done Preprocess data")
df.head()
# %%
# Vẽ biểu đồ giá đóng cửa
plt.figure(figsize=(10, 5))
plt.plot(df['close'], label="Market")
plt.title("Close price")
plt.xlabel("Time")
plt.ylabel("USD")
plt.legend()
plt.show()
# %%
# Xử lý dữ liệu
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))
cols_x = ['H-L', 'O-C', f'SMA_{ma_1}', f'SMA_{ma_2}',
          f'SMA_{ma_3}', f'SD_{ma_1}', f'SD_{ma_3}']
cols_y = ['close']
# Chuẩn hóa về khoảng 0-1
scaled_data_x = scaler_X.fit_transform(
    df[cols_x].values.reshape(-1, len(cols_x)))
scaled_data_y = scaler_Y.fit_transform(
    df[cols_y].values.reshape(-1, len(cols_y)))
# %%
# generate the input and output sequences
n_lookback = 60  # length of input sequences (lookback period)
n_forecast = 30  # length of output sequences (forecast period)

x_total = []
y_total = []

for i in range(n_lookback, len(df) - n_forecast):
    x_total.append(scaled_data_x[i - n_lookback: i])
    y_total.append(scaled_data_y[i: i + n_forecast])

x_total = np.array(x_total)
y_total = np.array(y_total)
print(x_total.shape[1], x_total.shape[2])
# %%
# Phân chia dữ liệu thành 3 phần train, validate, test
test_rate = 0.368
test_size = int(len(x_total) * test_rate)

x_train = np.array(x_total[:-test_size])
y_train = np.array(y_total[:-test_size])

x_validate = np.array(x_total[- test_size*2: -test_size])
y_validate = np.array(y_total[- test_size*2: -test_size])

x_test = np.array(x_total[-test_size:])
y_test = np.array(y_total[-test_size:])
print('Done process data: test_size: {}/{}\n'.format(test_size, len(x_total)))
print(x_train.shape, y_train.shape, x_test.shape,
      y_test.shape, x_validate.shape, y_validate.shape)
# %%
# Build Model
model = Sequential()  # Lớp mạng cho dữ liệu đầu vào
# Lớp LSTM 1 kết nối đầu vào, cần mô tả thông tin của đầu vào
model.add(LSTM(units=128, return_sequences=True,
          input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(units=64))  # Lớp LSTM 2
# Bỏ qua một số đơn vị ngẫu nhiên để tránh mô hình học tủ
model.add(Dropout(0.5))
model.add(Dense(n_forecast))  # Out put là giá đóng của của n_forecast ngày
# Sử dụng hàm đo sai số tuyệt đối trung bình và trình tối ưu hóa adam
model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=['mean_absolute_error'])
print("Build model success!")
# %%
# Train model
best_model = ModelCheckpoint(
    "model.h5", monitor="loss", verbose=2, save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=2,
          validation_data=(x_validate, y_validate), callbacks=[best_model])
print("Done Training Model")
# %%
# Sử dụng mô hình để dự đoán kết quả
plot_dates = dates[n_lookback:len(df)-n_forecast]
final_model = load_model("model.h5")

# Visualize prediction with train data
train_predictions = final_model.predict(x_train).flatten()
train_predictions = train_predictions.reshape(-1, len(cols_y))
dates_train = plot_dates[:-test_size]
len(dates_train)
plt.figure(figsize=(10, 5))
plt.plot(dates_train, scaler_Y.inverse_transform(train_predictions[::30]))
plt.plot(dates_train, scaler_Y.inverse_transform(y_train.reshape(-1, 1)[::30]))
plt.legend(['Training Predictions', 'Market'])
# %%
# Visualize prediction with test data
test_predictions = final_model.predict(x_test).flatten()
test_predictions = test_predictions.reshape(-1, len(cols_y))
dates_test = dates[-test_size:]
plt.figure(figsize=(10, 5))
plt.plot(dates_test, scaler_Y.inverse_transform(test_predictions[::30]))
plt.plot(dates_test, scaler_Y.inverse_transform(y_test.reshape(-1, 1)[::30]))
plt.legend(['Testing Predictions', 'Market'])
# %%
# Predict the stock next 30 days
x_predict = df[-n_lookback:][cols_x].values.reshape(-1, len(cols_x))
x_predict = scaler_X.fit_transform(x_predict)
x_predict = np.array(x_predict)
x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))
prediction = model.predict(x_predict)
prediction = scaler_Y.inverse_transform(prediction)
# print(prediction)

# organize the results in a data frame
df_past = df[['close']].reset_index()
df_past.rename(columns={'index': 'date', 'close': 'actual'}, inplace=True)
df_past['date'] = pd.to_datetime(df_past['date'])
df_past['forecast'] = np.nan
df_past['forecast'].iloc[-1] = df_past['actual'].iloc[-1]

df_future = pd.DataFrame(columns=['date', 'actual', 'forecast'])
df_future['date'] = pd.date_range(
    start=df_past['date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['forecast'] = prediction.flatten()
df_future['actual'] = np.nan

results = df_past.append(df_future).set_index('date')

# plot the results
plt.figure(figsize=(24, 8))
plt.plot(results)
plt.title('AAPL')
# %%
print("Train suitable rate:", r2_score(
    y_train.reshape(-1, 1), train_predictions))
print("Mean absolute train error:", mean_absolute_error(
    y_train.reshape(-1, 1), train_predictions))
print("")
print("Test suitable rate:", r2_score(y_test.reshape(-1, 1), test_predictions))
print("Mean absolute test error:", mean_absolute_error(
    y_test.reshape(-1, 1), test_predictions))
print("")
print("Mean absolute percentage error:", mean_absolute_percentage_error(
    y_test.reshape(-1, 1), test_predictions))
# %%
