# pip install streamlit prophet yfinance plotly
import streamlit as st
import pandas as pd
import numpy as np
# import yfinance as yf
# from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler  # Chuẩn hóa dữ liệu
from plotly import graph_objs as go
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
        st.sidebar.subheader('Query parameters')
        start_date = st.sidebar.date_input(
            "Start date", dt.date(2019, 1, 1))
        end_date = st.sidebar.date_input("End date", dt.datetime.now())
        # Retrieving tickers data
        ticker_list = pd.read_csv('company.csv')
        ticker_list = np.array(ticker_list['Ticker'])
        tickerSymbol = st.sidebar.selectbox(
            'Stock ticker', ticker_list)  # Select ticker symbol
        authenticator.logout("Logout", "sidebar")

    st.title('Stock Forecast App')
    tickerSymbol

    @st.cache
    def load_data(ticker):

        key = '3848aa6f3355ce8a788bdd508d862b26358c1963'
        df = pdr.get_data_tiingo(
            ticker, start=start_date, end=end_date, api_key=key)
        df.to_csv("data.csv")
        df = pd.read_csv("data.csv")
        df.reset_index(inplace=True)
        return df

    data_load_state = st.text('Loading data...')
    data = load_data(tickerSymbol)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data)

    # Plot raw data

    def plot_raw_data():
        fig = go.Figure(data=[go.Candlestick(x=data['date'],
                                             open=data['open'], high=data['high'],
                                             low=data['low'], close=data['close'])])
        fig.layout.update(
            title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet.

    def str_to_datetime(s):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return dt.datetime(year=year, month=month, day=day)
    data["date"] = pd.to_datetime(data.date, format="%Y-%m-%d %H:%M:%S")
    data["date"] = data["date"].dt.strftime('%Y-%m-%d')
    data['date'] = data['date'].apply(str_to_datetime)
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

    cols_x = ['H-L', 'O-C', f'SMA_{ma_1}', f'SMA_{ma_2}',
              f'SMA_{ma_3}', f'SD_{ma_1}', f'SD_{ma_3}']
    x_predict = df[-60:][cols_x].values.reshape(-1, len(cols_x))
    x_predict = np.array(x_predict)
    x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))
    prediction = model.predict(x_predict)

    df_past = df[['close']].reset_index()
    df_past.rename(columns={'index': 'date', 'close': 'actual'}, inplace=True)
    df_past['date'] = data['date']
    df_past['forecast'] = np.nan
    df_past['forecast'].iloc[-1] = df_past['actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['date', 'actual', 'forecast'])
    df_future['date'] = pd.date_range(
        start=data['date'].iloc[-1], periods=30)
    df_future['forecast'] = prediction.flatten()
    df_future['actual'] = np.nan

    results = df_past.append(df_future).set_index('date')
    st.write(results)
    st.line_chart(results)
