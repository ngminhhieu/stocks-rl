import numpy as np
from data.data_loader import LSTMDataLoader
import torch

class Environment():
    def __init__(self, config):
        self._config = config
        self._tp_score = config["model"]["tp_score"]
        self._tn_score = config["model"]["tn_score"]
        self._fp_score = config["model"]["fp_score"]
        self._fn_score = config["model"]["fn_score"]
        self._buy = 1
        self._sell = 0
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
        return self._list_state[0]

    def step(self, action):
        action_gt = self._gt[0][-1]
        self._list_state.pop(0)
        self._gt.pop(0)
        if action == self._buy:            
            if self.is_match(action, action_gt):
                score = self._tp_score
                self._matched['tp'] += 1 
            elif not self.is_match(action, action_gt):
                score = self._fp_score
                self._matched['fp'] += 1 
        elif action == self._sell:
            if self.is_match(action, action_gt):
                score = self._fn_score
                self._matched['fn'] += 1 
            elif not self.is_match(action, action_gt):
                score = self._tn_score
                self._matched['tn'] += 1
        else:
            raise ValueError(
                "Received invalid action={} which is not part of the action space".format(action))

        if len(self._list_state) == 0:
            next_state = (None, None)
            done = True
        else:
            next_state = self._list_state[0]
            done = False

        return next_state, score, done

    def is_match(self, action, action_gt):
        return action == action_gt