import numpy as np
import tensorflow as tf
import pandas as pd
from stockstats import StockDataFrame as Sdf
from utils.common import get_data_by_symbol_and_date, get_data_by_dataframe
import utils.indicators as indi
import torch

class DataLoader():
    def __init__(self, config):
        self._symbols = config["data"]["symbols"]
        self._config = config
        self._his_window = config["data"]["history_window"]
        self._indicators = config["data"]["indicators"]
        self._include_target = config["data"]["include_target"]
        self._target_col = config["data"]["target_col"]
        self._max_slow_period = max([max(indi.values()) for indi in self._indicators.values() if indi.values()])
        self._n_step_ahead = config["data"]["n_step_ahead"]
        self._outlier_threshold = config["data"]["outlier_threshold"]


    def gen_backtest_data_multi_seq(self, start_date="2021-01-11", end_date='2021-11-18'):
        gts = {}
        inputs = []
        inputs_storage = []
        for sym in self._symbols:
            data = get_data_by_symbol_and_date(sym, start_date, end_date, self._his_window + self._max_slow_period)
            # data = get_data_by_dataframe(sym, start_date, end_date, self._his_window + self._max_slow_period)
            inputs_storage.append(data)
        inputs_storage = self._fill_missing_data(inputs_storage)
        for ind in range(len(inputs_storage)):
            data = inputs_storage[ind]
            df, df_x, _ = self._preprocess_data(data, self._indicators)
            df = df.iloc[self._his_window:]
            df_x = df_x.to_numpy()
            gts[self._symbols[ind]] = df[["open", "high", "low", "close", "volume"]]
            input_data = np.array([df_x[i: i + self._his_window]
                                for i in range(len(df_x) - self._his_window)])
            inputs.append(input_data)
        inputs = np.dstack((inputs))
        return inputs, gts


    def split_train_test_by_date_cls_prob(self, start_train, end_train, start_test, end_test):
        X_train_storage = []
        X_test_storage = []
        y_train_storage = []
        y_test_storage = []
        train_data_storage = []
        test_data_storage = []
        print("Downloading data...")
        for sym in self._symbols:
            train_data = get_data_by_symbol_and_date(sym, start_train, end_train, self._his_window + self._max_slow_period)
            # train_data = get_data_by_dataframe(sym, start_train, end_train, self._his_window + self._max_slow_period)
            train_data_storage.append(train_data)
            test_data = get_data_by_symbol_and_date(sym, start_test, end_test, self._his_window + self._max_slow_period)
            # test_data = get_data_by_dataframe(sym, start_test, end_test, self._his_window + self._max_slow_period)
            test_data_storage.append(test_data)
        
        train_data_storage = self._fill_missing_data(train_data_storage)
        test_data_storage = self._fill_missing_data(test_data_storage)

        print("Processing data...")
        for ind in range(len(train_data_storage)):
            train_data = train_data_storage[ind]
            test_data = test_data_storage[ind]
            _, df_x_train, df_y_train = self._preprocess_data(train_data, self._indicators)
            _, df_x_test, df_y_test = self._preprocess_data(test_data, self._indicators)
            
            train_x = df_x_train.to_numpy()
            train_y = df_y_train.to_numpy()
            test_x = df_x_test.to_numpy()
            test_y = df_y_test.to_numpy()

            X_train = []
            y_train = []
            X_test = []
            y_test = []

            X_train = np.array([train_x[i: i + self._his_window]
                                for i in range(len(train_x)-self._his_window)])
            y_train = np.array([train_y[i + self._his_window]
                                for i in range(len(train_y)-self._his_window)])
            X_test = np.array([test_x[i: i + self._his_window]
                            for i in range(len(test_x)-self._his_window)])
            y_test = np.array([test_y[i + self._his_window]
                            for i in range(len(test_y)-self._his_window)])
            X_train_storage.append(X_train)
            X_test_storage.append(X_test)
            y_train_storage.append(y_train)
            y_test_storage.append(y_test)

        X_train_full = np.dstack((X_train_storage))
        y_train_full = np.dstack((y_train_storage))
        y_train_full = np.reshape(y_train_full, (y_train_full.shape[0], y_train_full.shape[2], y_train_full.shape[1]))
        X_test_full = np.dstack((X_test_storage))
        y_test_full = np.dstack((y_test_storage))
        y_test_full = np.reshape(y_test_full, (y_test_full.shape[0], y_test_full.shape[2], y_test_full.shape[1]))
        
        return X_train_full, y_train_full, X_test_full, y_test_full


    def _preprocess_data(self, source_df, indicators):
        df = source_df.copy()
        features = []

        if "close_ratio" in indicators:
            medium_period = indicators["close_ratio"]["medium_period"]
            slow_period = indicators["close_ratio"]["slow_period"]
            df, features = indi.close_ratio(df, features, medium_period, slow_period)

        if "volume_ratio" in indicators:
            medium_period = indicators["volume_ratio"]["medium_period"]
            slow_period = indicators["volume_ratio"]["slow_period"]
            df, features = indi.volume_ratio(df, features, medium_period, slow_period)

        if "close_sma" in indicators:
            medium_period = indicators["close_sma"]["medium_period"]
            slow_period = indicators["close_sma"]["slow_period"]
            df, features = indi.close_sma(df, features, medium_period, slow_period)

        if "volume_sma" in indicators:
            medium_period = indicators["volume_sma"]["medium_period"]
            slow_period = indicators["volume_sma"]["slow_period"]
            df, features = indi.volume_sma(df, features, medium_period, slow_period)
        
        if "close_ema" in indicators:
            medium_period = indicators["close_ema"]["medium_period"]
            slow_period = indicators["close_ema"]["slow_period"]
            df, features = indi.close_sma(df, features, medium_period, slow_period)

        if "volume_ema" in indicators:
            medium_period = indicators["volume_ema"]["medium_period"]
            slow_period = indicators["volume_ema"]["slow_period"]
            df, features = indi.volume_sma(df, features, medium_period, slow_period)

        if "atr" in indicators:
            medium_period = indicators["atr"]["medium_period"]
            slow_period = indicators["atr"]["slow_period"]
            df, features = indi.atr(df, features, medium_period, slow_period)

        if "adx" in indicators:
            medium_period = indicators["adx"]["medium_period"]
            slow_period = indicators["adx"]["slow_period"]
            df, features = indi.adx(df, features, medium_period, slow_period)
        
        if "kdj" in indicators:    
            medium_period = indicators["kdj"]["medium_period"]
            slow_period = indicators["kdj"]["slow_period"]
            df, features = indi.kdj(df, features, medium_period, slow_period)

        if "rsi" in indicators:  
            medium_period = indicators["rsi"]["medium_period"]
            slow_period = indicators["rsi"]["slow_period"]          
            df, features = indi.rsi(df, features, medium_period, slow_period)

        if "macd" in indicators:    
            medium_period = indicators["macd"]["medium_period"]
            slow_period = indicators["macd"]["slow_period"]        
            df, features = indi.macd(df, features, medium_period, slow_period)
        
        if "bb" in indicators:                
            df, features = indi.bb(df, features)

        if "arithmetic_returns" in indicators:
            df, features = indi.arithmetic_returns(df, features)
        
        if "obv" in indicators:
            medium_period = indicators["rsi"]["medium_period"]
            slow_period = indicators["rsi"]["slow_period"]
            df, features = indi.obv(df, features, medium_period, slow_period)

        if "mfi" in indicators:
            medium_period = indicators["rsi"]["medium_period"]
            slow_period = indicators["rsi"]["slow_period"]
            df, features = indi.mfi(df, features, medium_period, slow_period)

        if "ichimoku" in indicators:
            fast_period = indicators["ichimoku"]["fast_period"]
            medium_period = indicators["ichimoku"]["medium_period"]
            slow_period = indicators["ichimoku"]["slow_period"]        
            df, features = indi.ichimoku(df, features, fast_period, medium_period, slow_period)
        
        if "k_line" in indicators:
            df, features = indi.k_line(df, features)
        
        if "eight_trigrams" in indicators:
            df, features = indi.eight_trigrams(df, features)
        
        if "trend_return" in indicators:
            df, features = indi.trend_return(df, features, self._n_step_ahead)
        
        if "trend" in indicators: 
            trend_up_threshold = indicators["trend"]["trend_up_threshold"]
            trend_down_threshold = indicators["trend"]["trend_down_threshold"]
            df, features = indi.trend(df, features, trend_up_threshold, trend_down_threshold, self._n_step_ahead)

        df = indi.remove_outliers(df, features, threshold = self._outlier_threshold)
        df = df[slow_period:-self._n_step_ahead] # slow_period: SMA and other indicators, 5: days of a week
        if not self._include_target and self._target_col in features:
            features.remove(self._target_col)
        df_y = df[self._target_col]
        df_x = df.filter((features))
        return df, df_x, df_y


    def _fill_missing_data(self, list_df):
        full_data = max(list_df, key=len)
        max_len = len(full_data) # Get max len of the symbol having full data
        for i in range(len(list_df)):
            if len(list_df[i]) < max_len:
                for count, ind in enumerate(full_data.index):
                    if ind not in list_df[i].index:
                        list_df[i].loc[ind] = list_df[i].loc[full_data.iloc[count-1].name].copy()
                        list_df[i].loc[ind]["open"] = list_df[i].loc[ind]["close"].copy()
            list_df[i].sort_index(inplace=True)
        
        return list_df