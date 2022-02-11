import requests
import pandas as pd
from datetime import datetime, timedelta

def format_str_datetime(str_date):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return date.strftime('%Y-%m-%d')

def add_days_to_string_time(str_date, days=1):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    new_date = date + timedelta(days=days)
    return new_date.strftime('%Y-%m-%d')

def _get_start_date_with_offset(start_date, offset):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')

    weekdays = [0, 1]
    count = 0
    days = timedelta(1)
    while count != offset:
        start_date = start_date - days
        if start_date.weekday() not in weekdays: 
            count += 1
    return start_date.strftime('%Y-%m-%d')
        

def generate_date_index(start_date, periods):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    list_index = []
    weekdays = [0, 1]
    count = 0
    days = timedelta(1)
    while count != periods:
        start_date = start_date + days
        if start_date.weekday() not in weekdays: 
            count += 1
            list_index.append(start_date.strftime('%Y-%m-%d'))
    return list_index


def days_between(start_time, end_time):
    d1 = str(pd.to_datetime(start_time))
    d2 = str(pd.to_datetime(end_time))
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    return (d2-d1).days


def get_data_by_symbol_and_date(symbol, start_date, end_date, offset=0):
    if offset:
        start_date = _get_start_date_with_offset(start_date, offset)
    params = {
        "ticker": symbol,
        "start_date": start_date,
        "end_date": end_date
    }
    res = requests.post("http://192.168.67.129:9997/data/daily_history", params=params)
    # res = requests.post("http://202.191.57.62:9997/data/daily_history", params=params)
    data = res.json()
    if 'data' in data:
        data = data["data"]
    else:
        print("Error when fetching data!")
        exit()
    df_data = pd.DataFrame(data)
    df_data.index = pd.to_datetime(df_data['date'])
    df_data.index = df_data.index.map(str)
    df_data = df_data.drop(columns=['symbol', 'date'])
    df_data = df_data.astype(float)
    return df_data


def get_data_by_dataframe(symbol, start_date, end_date, offset=0):
    if offset:
        start_date = _get_start_date_with_offset(start_date, offset)
        
    df_data = pd.read_csv("data/US/SP500/{}_1day.csv".format(symbol), index_col="Date")
    df_data = df_data[df_data.index >= start_date]
    df_data = df_data[df_data.index <= end_date]
    df_data["high"] = df_data["High"]
    df_data["low"] = df_data["Low"]
    df_data["open"] = df_data["Open"]
    df_data["close"] = df_data["Close"]
    df_data["volume"] = df_data["Volume"]
    df_data.index = df_data.index.map(str)
    df_data = df_data.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
    df_data = df_data.astype(float)
    return df_data