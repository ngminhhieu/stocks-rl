import numpy as np
from data.data_loader import LSTMDataLoader

class Environment():
    def __init__(self, config):
        self._config = config
        self._tp_score = config["model"]["tp_score"]
        self._tn_score = config["model"]["tn_score"]
        self._fp_score = config["model"]["fp_score"]
        self._fn_score = config["model"]["fn_score"]
        self._hold_signal = 0
        self._buy_signal = 0
        self._sell_signal = 1
        self._matched = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

        self._start_train = config["data"]["start_train"]
        self._end_train = config["data"]["end_train"]
        self._start_backtest = config["data"]["start_backtest"]
        self._end_backtest = config["data"]["end_backtest"]
        dataloader = LSTMDataLoader(config)
        self._X_train, self._y_train, self._X_test, self._y_test = dataloader.split_train_test(
            self._start_train, self._end_train, self._start_backtest, self._end_backtest)
        self._X_train = list(np.squeeze(self._X_train, 1))
        self._X_test = list(np.squeeze(self._X_test, 1))
        self._y_train = list(self._y_train)
        self._y_test = list(self._y_test)

        self._list_state = self._X_train.copy()
        self._gt = self._y_train.copy()

        self._reward = 0
        self._balance = config["model"]["initial_balance"]
        self._initial_balance = config["model"]["initial_balance"]
        self._total_asset = self._balance
        self._commission = config["model"]["commission"]
        self._current_size = 0
        self._total_asset = 0
    

    def get_num_fts(self):
        return self._X_train[0].shape[-1]


    def reset(self):
        """
          Important: The observation must be numpy array 
          : return: (np.array)
        """
        self._list_state = self._X_train.copy()
        self._gt = self._y_train.copy()
        self._matched = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        self._balance = self._config["model"]["initial_balance"]
        self._current_size = 0
        self._total_asset = self._balance
        self._reward = 0
        return self._list_state[0]

    def step(self, action, size=1000):
        close_price = self._gt[0][-1]
        self._list_state.pop(0)
        self._gt.pop(0)
        if action == self._buy_signal:
            self._buy(close_price, size)
                
        elif action == self._sell_signal:
            self._sell(close_price, size)

        elif action == self._hold_signal:
            pass

        else:
            raise ValueError(
                "Received invalid action={} which is not part of the action space".format(action))

        if len(self._list_state) == 0:
            next_state = (None, None)
            done = True
        else:
            next_state = self._list_state[0]
            self._total_asset = self._balance + (self._current_size * close_price)
            self._reward = self._total_asset / self._initial_balance
            done = False

        return next_state, self._reward, done


    def _buy(self, close_price, size):
        if self._balance > close_price * size * (1 + self._commission):
            buy_value = close_price * size
            commission = buy_value * self._commission
            self._current_size += size
            self._balance -= buy_value + commission # balance - buy_value - commission

    def _sell(self, close_price, size):
        if self._current_size > size:
            sell_value = close_price * size
            commission = sell_value * self._commission
            self._current_size -= size
            self._balance += sell_value - commission # balance + sell_value - commission