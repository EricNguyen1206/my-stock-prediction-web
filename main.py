# pip install streamlit prophet yfinance plotly
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import pickle
from pathlib import Path

import datetime as dt
import tensorflow as tf
import pandas_datareader as pdr

import streamlit_authenticator as stauth  # pip install streamlit-authenticator

key = '3848aa6f3355ce8a788bdd508d862b26358c1963'

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
    }}, "sales_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    # ---- SIDEBAR ----
    with st.sidebar:
        st.sidebar.title(f"Welcome {name}")
        add_radio = st.radio(
            "Choose a shipping method",
            ("Standard (5-15 days)", "Express (2-5 days)")
        )
        authenticator.logout("Logout", "sidebar")

    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.title('Stock Forecast App')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # @st.cache
    # def load_data_tiingo(ticker):
    #     df = pdr.get_data_tiingo(ticker, start=START, end=TODAY, api_key=key)
    #     df.reset_index(inplace=True)
    #     return df

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    # data_load_state = st.text('Loading data...')
    # datatiingo = load_data_tiingo(selected_stock)
    # data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data)

    # st.subheader('Raw data tiingo')
    # st.write(datatiingo)

    # Plot raw data

    def plot_raw_data():
        fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                             open=data['Open'], high=data['High'],
                                             low=data['Low'], close=data['Close'])])
        fig.layout.update(
            title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail(forecast.shape[0]-len(df_train)))

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
