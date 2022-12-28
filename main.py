# pip install streamlit prophet plotly
import streamlit as st
import pandas as pd
import numpy as np
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler  # Chuẩn hóa dữ liệu
from plotly import graph_objs as go
from plotly import express as px
from plotly import figure_factory as ff
from keras.models import load_model  # tải mô hình

import pickle
from pathlib import Path

import datetime as dt
import pandas_datareader as pdr

import streamlit_authenticator as stauth  # pip install streamlit-authenticator

# --- USER AUTHENTICATION ---
names = ["Peter Parker", "Rebecca Miller"]
usernames = ["pparker", "rmiller"]

# load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(
    {'usernames': {
        usernames[0]: {'name': names[0], 'password': hashed_passwords[0]},
        usernames[1]: {'name': names[1], 'password': hashed_passwords[1]},
    }}, "sales_dashboard", "abcdef", cookie_expiry_days=1)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    # ---- SIDEBAR ----
    with st.sidebar:
        st.sidebar.title(f"Welcome {name}")
        st.sidebar.subheader('Categorize data')

        start_date = st.sidebar.date_input(
            "Start date", value=dt.datetime.now() - dt.timedelta(150), max_value=dt.datetime.now() - dt.timedelta(150))
        end_date = st.sidebar.date_input(
            "End date", dt.datetime.now(), min_value=start_date+dt.timedelta(150))
        # Retrieving tickers data
        ticker_list = pd.read_csv('company.csv')
        ticker_list = np.array(ticker_list['Ticker'])
        tickerSymbol = st.sidebar.selectbox(
            'Stock ticker', ticker_list)  # Select ticker symbol
        authenticator.logout("Logout", "sidebar")

    st.title('Stock Forecast App')
    tickerSymbol

    @st.cache(allow_output_mutation=True)
    def load_data(ticker):
        key = '3848aa6f3355ce8a788bdd508d862b26358c1963'
        df = pdr.get_data_tiingo(
            ticker, start=start_date, end=end_date, api_key=key)
        return df

    data_load_state = st.text('Loading data...')
    data = load_data(tickerSymbol)
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)
    data_load_state.text('Loading data... done!')

    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return dt.datetime(year=year, month=month, day=day)
    data["date"] = pd.to_datetime(data.date, format="%Y-%m-%d %H:%M:%S")
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")

    st.subheader("Raw data")
    st.write(data.iloc[::-1])

    # Plot raw data

    # data["date"] = data["date"].apply(str_to_datetime)

    def plot_raw_data():
        fig = go.Figure(data=[go.Candlestick(x=data['date'],
                                             open=data['open'], high=data['high'],
                                             low=data['low'], close=data['close'])])
        fig.layout.update(
            title_text=f'Candlestick chart of {tickerSymbol}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Preprocess data
    df = data[['close', 'high', 'low', 'open']]
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
    df.dropna(inplace=True)
    model = load_model("model.h5")
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    cols_x = ['H-L', 'O-C', f'SMA_{ma_1}', f'SMA_{ma_2}',
              f'SMA_{ma_3}', f'SD_{ma_1}', f'SD_{ma_3}']
    cols_y = ['close']

    scaled_data_x = scaler_X.fit_transform(
        df[cols_x].values.reshape(-1, len(cols_x)))
    scaled_data_y = scaler_Y.fit_transform(
        df[cols_y].values.reshape(-1, len(cols_y)))

    x_predict = scaled_data_x[-60:]
    x_predict = np.array(x_predict)
    x_predict = scaler_X.fit_transform(x_predict)
    x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))
    prediction = model.predict(x_predict)
    prediction = scaler_Y.inverse_transform(prediction)

    df_past = data[['close']].reset_index()
    df_past.rename(columns={'index': 'date', 'close': 'market'}, inplace=True)
    df_past['date'] = data[['date']]
    df_past.set_index('date')
    df_past['forecast'] = np.nan
    # df_past['forecast'].iloc[-1] = df_past['market'].iloc[-1]

    df_future = pd.DataFrame(columns=['date', 'market', 'forecast'])
    df_future['date'] = pd.date_range(
        start=dt.date.today(), periods=30)
    df_future['date'] = df_future['date'].dt.strftime("%Y-%m-%d")
    df_future['forecast'] = prediction.flatten()
    df_future['market'] = np.nan

    results = df_past.tail(60).append(df_future).set_index('date')
    st.subheader("Forecast")
    # results.to_csv('results.csv')
    # st.line_chart(results)

    def plot_forecast_data():
        fig = px.line(results, x=results.index, y=[
                      'market', 'forecast'], title=f'Close Price forecast of {tickerSymbol}')
        fig.layout.update(
            title_text=f'Close Price forecast of {tickerSymbol}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_forecast_data()

    # st.write(df_future['forecast'].values[-1]-df_future['forecast'].values[0])
    def get_roi(val_begin, val_end):
        return (val_end - val_begin)/val_begin
    roi = get_roi(df_future['forecast'].values[0],
                  df_future['forecast'].values[-1])

    def plot_roi_normal_distribution():
        df_future['percent_roi'] = np.nan
        for i in range(1, len(df_future['forecast'])):
            df_future.at[i, 'percent_roi'] = 100*get_roi(
                df_future['forecast'].values[i-1], df_future['forecast'].values[i])
        df_future.at[0, 'percent_roi'] = 100*get_roi(
            df_past['market'].values[-1], df_future['forecast'].values[0])
        st.write(df_future[['date', 'forecast', 'percent_roi']])
        group_labels = ['percent_roi']
        fig = ff.create_distplot(
            [df_future['percent_roi']], group_labels, curve_type='normal')
        fig.layout.update(
            title_text=f'Normal distribution ROI percent of {tickerSymbol}', xaxis_rangeslider_visible=True, template='plotly_dark')
        st.plotly_chart(fig)

    plot_roi_normal_distribution()
    st.subheader("Summary")
    st.write(f"Return on Investment rate next 30 days: {round(roi, 6)}")
    st.write(
        f"Return on Investment percent next 30 days: {round(roi*100, 3)} %")
