import numpy as np
from utils.common import get_data_by_symbol_and_date
from utils.data_loader import DataLoader


class LSTMDataLoader(DataLoader):
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


    def split_train_test(self, start_train, end_train, start_test, end_test):
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
            y_train = np.array([train_y[i + 1: i + self._his_window + 1]
                                for i in range(len(train_y)-self._his_window)])
            X_test = np.array([test_x[i: i + self._his_window]
                            for i in range(len(test_x)-self._his_window)])
            y_test = np.array([test_y[i + 1: i + self._his_window + 1]
                            for i in range(len(test_y)-self._his_window)])
            
            X_train_storage.append(X_train)
            X_test_storage.append(X_test)
            y_train_storage.append(y_train)
            y_test_storage.append(y_test)

        num_train_sample = X_train_storage[0].shape[0]
        X_train_full = np.vstack((X_train_storage))
        X_train_full = np.reshape(X_train_full, (num_train_sample, len(self._symbols), X_train_full.shape[1], X_train_full.shape[2]))
        y_train_full = np.vstack((y_train_storage))
        y_train_full = np.reshape(y_train_full, (num_train_sample, y_train_full.shape[1], len(self._symbols)))

        num_test_sample = X_test_storage[0].shape[0]
        X_test_full = np.vstack((X_test_storage))
        X_test_full = np.reshape(X_test_full, (num_test_sample, len(self._symbols), X_test_full.shape[1], X_test_full.shape[2]))
        y_test_full = np.vstack((y_test_storage))
        y_test_full = np.reshape(y_test_full, (num_test_sample, y_test_full.shape[1], len(self._symbols)))
        
        return X_train_full, y_train_full, X_test_full, y_test_full


    def gen_backtest_data_multi_seq(self, start_date, end_date):
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
        num_sample = inputs[0].shape[0]
        inputs = np.vstack((inputs))
        inputs = np.reshape(inputs, (num_sample, len(self._symbols), inputs.shape[1], inputs.shape[2]))
        return inputs, gts