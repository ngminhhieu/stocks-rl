from pyexpat import features
from stockstats import StockDataFrame as Sdf
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import IchimokuIndicator

def close_ratio(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["close_short_max".format(medium_period)] = stockstat["close_-{}~0_max".format(medium_period)]
    df["close_short_max_ratio".format(medium_period)] = stockstat["close"] / stockstat["close_-{}~0_max".format(medium_period)] - 1
    df["close_long_max".format(slow_period)] = stockstat["close_-{}~0_max".format(slow_period)]
    df["close_long_max_ratio".format(medium_period)] = stockstat["close"] / stockstat["close_-{}~0_max".format(slow_period)] - 1
    df["close_short_min".format(medium_period)] = stockstat["close_-{}~0_min".format(medium_period)]
    df["close_short_min_ratio".format(medium_period)] = stockstat["close"] / stockstat["close_-{}~0_min".format(medium_period)] - 1
    df["close_long_min".format(slow_period)] = stockstat["close_-{}~0_min".format(slow_period)]
    df["close_long_min_ratio".format(medium_period)] = stockstat["close"] / stockstat["close_-{}~0_min".format(slow_period)] - 1

    features.extend(["close_short_max_ratio", "close_long_max_ratio", "close_short_min_ratio", "close_long_min_ratio"])
    return df, features

def volume_ratio(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["volume_short_max".format(medium_period)] = stockstat["volume_-{}~0_max".format(medium_period)]
    df["volume_short_max_ratio".format(medium_period)] = stockstat["volume"] / stockstat["volume_-{}~0_max".format(medium_period)] - 1
    df["volume_long_max".format(slow_period)] = stockstat["volume_-{}~0_max".format(slow_period)]
    df["volume_long_max_ratio".format(medium_period)] = stockstat["volume"] / stockstat["volume_-{}~0_max".format(slow_period)] - 1
    df["volume_short_min".format(medium_period)] = stockstat["volume_-{}~0_min".format(medium_period)]
    df["volume_short_min_ratio".format(medium_period)] = stockstat["volume"] / stockstat["volume_-{}~0_min".format(medium_period)] - 1
    df["volume_long_min".format(slow_period)] = stockstat["volume_-{}~0_min".format(slow_period)]
    df["volume_long_min_ratio".format(medium_period)] = stockstat["volume"] / stockstat["volume_-{}~0_min".format(slow_period)] - 1
    
    features.extend(["volume_short_max_ratio", "volume_long_max_ratio", "volume_short_min_ratio", "volume_long_min_ratio"])
    return df, features

def close_sma(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["close_short_sma"] = stockstat["close_{}_sma".format(medium_period)]
    df["close_long_sma"] = stockstat["close_{}_sma".format(slow_period)]
    df["short_long_close_sma_ratio"] = df["close_short_sma"] / df["close_long_sma"] - 1
    df["long_close_sma_ratio"] = df["close"] / df["close_long_sma"] - 1
    df["short_close_sma_ratio"] = df["close"] / df["close_short_sma"] - 1

    features.extend(["short_long_close_sma_ratio", "long_close_sma_ratio", "short_close_sma_ratio"])
    return df, features

def volume_sma(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["volume_short_sma"] = stockstat["volume_{}_sma".format(medium_period)]
    df["volume_long_sma"] = stockstat["volume_{}_sma".format(slow_period)]
    df["short_long_volume_sma_ratio"] = df["volume_short_sma"] / df["volume_long_sma"] - 1
    df["long_volume_sma_ratio"] = df["volume"] / df["volume_long_sma"] - 1
    df["short_volume_sma_ratio"] = df["volume"] / df["volume_short_sma"] - 1
    
    features.extend(["short_long_volume_sma_ratio", "long_volume_sma_ratio", "short_volume_sma_ratio"])
    return df, features

def close_ema(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["close_short_ema"] = stockstat["close_{}_ema".format(medium_period)]
    df["close_long_ema"] = stockstat["close_{}_ema".format(slow_period)]
    df["short_long_close_ema_ratio"] = df["close_short_ema"] / df["close_long_ema"] - 1
    df["long_close_ema_ratio"] = df["close"] / df["close_long_ema"] - 1
    df["short_close_ema_ratio"] = df["close"] / df["close_short_ema"] - 1

    features.extend(["short_long_close_ema_ratio", "long_close_ema_ratio", "short_close_ema_ratio"])
    return df, features

def volume_ema(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["volume_short_ema"] = stockstat["volume_{}_ema".format(medium_period)]
    df["volume_long_ema"] = stockstat["volume_{}_ema".format(slow_period)]
    df["short_long_volume_ema_ratio"] = df["volume_short_ema"] / df["volume_long_ema"] - 1
    df["long_volume_ema_ratio"] = df["volume"] / df["volume_long_ema"] - 1
    df["short_volume_ema_ratio"] = df["volume"] / df["volume_short_ema"] - 1
    
    features.extend(["short_long_volume_ema_ratio", "long_volume_ema_ratio", "short_volume_ema_ratio"])
    return df, features

def atr(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["atr_short"] = stockstat["atr_{}".format(medium_period)]
    df["atr_long"] = stockstat["atr_{}".format(slow_period)]
    df["atr_ratio"] = df["atr_long"] / df["atr_short"] - 1
    
    features.extend(["atr_ratio"])
    return df, features

def adx(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["adx_short".format(medium_period)] = stockstat["dx_{}_ema".format(medium_period)] / 25 - 1        
    df["adx_long".format(slow_period)] = stockstat["dx_{}_ema".format(slow_period)] / 25 - 1        
    
    features.extend(["adx_short", "adx_long"])
    return df, features

def kdj(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["kdj_short"] = stockstat["kdjk_{}".format(medium_period)] / 50 - 1
    df["kdj_short_ratio"] = stockstat["kdjk_{}".format(medium_period)] / stockstat["kdjd_{}".format(medium_period)] - 1
    df["kdj_long"] = stockstat["kdjk_{}".format(slow_period)] / 50 - 1
    df["kdj_long_ratio"] = stockstat["kdjk_{}".format(slow_period)] / stockstat["kdjd_{}".format(slow_period)] - 1
    
    features.extend(["kdj_short", "kdj_short_ratio", "kdj_long", "kdj_long_ratio"])
    return df, features

def rsi(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    df["rsi_short"] = stockstat["rsi_{}".format(medium_period)] / 50 - 1
    df["rsi_long"] = stockstat["rsi_{}".format(slow_period)] / 50 - 1
    # df["rsi_ratio"] = df["rsi_long"] / df["rsi_short"] - 1
    
    features.extend(["rsi_short", "rsi_long"])
    return df, features

def macd(df, features, medium_period, slow_period):
    stockstat = Sdf.retype(df.copy())
    ema_short = 'close_{}_ema'.format(max(12, medium_period))
    ema_long = 'close_{}_ema'.format(min(26, slow_period))
    ema_signal = 'macd_{}_ema'.format(9)
    fast = stockstat[ema_short]
    slow = stockstat[ema_long]
    stockstat['macd'] = fast - slow
    stockstat['macd_sign'] = fast / slow - 1
    stockstat['macds'] = stockstat[ema_signal]
    stockstat['macdh'] = (stockstat['macd'] - stockstat['macds'])
    stockstat["macd_ratio"] = stockstat['macd'] / stockstat['macds'] - 1
    
    df['MACD_signal'] = stockstat["macd_ratio"]
    df['MACD_sign'] = stockstat["macd_sign"]
    features.extend(["MACD_sign", "MACD_signal"])
    return df, features
    

def bb(df, features):
    stockstat = Sdf.retype(df.copy())
    df['boll_lb'] = stockstat['boll_lb'] / df['close'] - 1
    df['boll_ub'] = stockstat['boll_ub'] / df['close'] - 1

    features.extend(["boll_lb", "boll_ub"])
    return df, features

def trend_return(df, features, n_step_ahead):
    df['daily_return'] = df['close'].pct_change()
    df['trend_return'] = df['close'].pct_change(periods=n_step_ahead) # n_step_ahead=5 is a week
    df['trend_return'] = df['trend_return'].shift(-n_step_ahead)

    features.extend(["trend_return"])
    return df, features

def trend(df, features, trend_up_threshold, trend_down_threshold, n_step_ahead):
    df['daily_return'] = df['close'].pct_change()
    df['trend_return'] = df['close'].pct_change(periods=n_step_ahead) # n_step_ahead=5 is a week
    df['trend_return'] = df['trend_return'].shift(-n_step_ahead)
    df["trend"] = 0
    df.loc[(df['trend_return'] > trend_up_threshold), 'trend'] = 1
    df.loc[(df['trend_return'] <= -trend_down_threshold), 'trend'] = 0

    features.extend(["trend"])
    return df, features

def arithmetic_returns(df, features):
    df['open_r'] = df['open'] / df['close'] - 1 
    df['high_r'] = df['high'] / df['close'] - 1
    df['low_r'] = df['low'] / df['close']  - 1 
    df['close_r'] = df['close'].pct_change()  
    df['volume_r'] = df['volume'].pct_change()

    features.extend(["open_r", "high_r", "low_r", "close_r", "volume_r"])
    return df, features

def mfi(df, features, medium_period, slow_period):
    mfi_short = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=medium_period)
    mfi_long = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=slow_period)
    df['mfi_short'] = mfi_short.money_flow_index()
    df['mfi_long'] = mfi_long.money_flow_index()
    df['mfi_short_ratio'] = df['mfi_short'] / 50 - 1
    df['mfi_long_ratio'] = df['mfi_long'] / 50 - 1
    # df["mfi_ratio"] = df["mfi_long"] / df["mfi_short"] - 1
    features.extend(["mfi_short_ratio", "mfi_long_ratio"])
    return df, features


def obv(df, features, medium_period, slow_period):
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    df['obv_short'] = df['obv'].pct_change(periods=medium_period)
    df['obv_long'] = df['obv'].pct_change(periods=slow_period)    
    features.extend(['obv_short', 'obv_long'])
    return df, features

def ichimoku(df, features, fast_period, medium_period, slow_period):
    ichimoku = IchimokuIndicator(df['high'], df['low'], fast_period, medium_period, slow_period)
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()

    features.extend(['ichimoku_conversion_line', 'ichimoku_base_line', 'ichimoku_a', 'ichimoku_b'])
    return df, features

# Stock Trend Prediction Using Candlestick Charting and Ensemble Machine Learning Techniques with a Novelty Feature Engineering Scheme
def k_line(df, features):
    df['k_line'] = 0
    msk_0 = (df['open'] == df['close']) & (df['open'] == df['low']) & (df['open'] == df['high'])
    msk_1 = (df['open'] == df['close']) & (df['open'] == df['high']) & (df['open'] == df['low'])
    msk_2 = (df['open'] == df['low']) & (df['close'] == df['high'])
    msk_3 = (df['open'] == df['high']) & (df['close'] == df['low'])
    msk_4 = (df['open'] == df['close']) & (df['open'] == df['high']) & (df['low'] < df['close'])
    msk_5 = (df['open'] == df['close']) & (df['open'] == df['low']) & (df['high'] > df['high'])
    msk_6 = (df['open'] == df['close']) & (df['low'] < df['close']) & (df['high'] > df['close'])
    msk_7 = (df['open'] > df['low']) & (df['close'] > df['open']) & (df['close'] == df['high'])
    msk_8 = (df['close'] > df['low']) & (df['open'] > df['close']) & (df['open'] == df['high'])
    msk_9 = (df['open'] == df['low']) & (df['close'] > df['open']) & (df['high'] > df['close'])
    msk_10 = (df['close'] == df['low']) & (df['open'] > df['close']) & (df['high'] > df['open'])
    msk_11 = (df['open'] < df['close']) & (df['low'] < df['open']) & (df['high'] > df['close'])
    msk_12 = (df['open'] > df['close']) & (df['low'] < df['close']) & (df['high'] > df['open'])
    
    df.loc[msk_0, 'k_line'] = 0
    df.loc[msk_1, 'k_line'] = 1
    df.loc[msk_2, 'k_line'] = 2
    df.loc[msk_3, 'k_line'] = 3
    df.loc[msk_4, 'k_line'] = 4
    df.loc[msk_5, 'k_line'] = 5
    df.loc[msk_6, 'k_line'] = 6
    df.loc[msk_7, 'k_line'] = 7
    df.loc[msk_8, 'k_line'] = 8
    df.loc[msk_9, 'k_line'] = 9
    df.loc[msk_10, 'k_line'] = 10
    df.loc[msk_11, 'k_line'] = 11
    df.loc[msk_12, 'k_line'] = 12
    features.extend(['k_line'])
    return df, features

def eight_trigrams(df, features):
    df['high_pre'] = df['high'].shift(1)
    df['low_pre'] = df['low'].shift(1)
    df['close_pre'] = df['close'].shift(1)
    df['open_pre'] = df['open'].shift(1)
    df['eight_trigrams'] = 0
    # bear high
    msk_0 = (df['high'] > df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] > df['low_pre'])
    msk_1 = (df['high'] < df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] < df['low_pre'])
    msk_2 = (df['high'] < df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] > df['low_pre'])
    msk_3 = (df['high'] > df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] > df['low_pre'])
    msk_4 = (df['high'] < df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] < df['low_pre'])
    msk_5 = (df['high'] > df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] < df['low_pre'])
    msk_6 = (df['high'] < df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] > df['low_pre'])
    msk_7 = (df['high'] > df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] < df['low_pre'])
    df.loc[msk_0, 'eight_trigrams'] = 0
    df.loc[msk_1, 'eight_trigrams'] = 1
    df.loc[msk_2, 'eight_trigrams'] = 2
    df.loc[msk_3, 'eight_trigrams'] = 3
    df.loc[msk_4, 'eight_trigrams'] = 4
    df.loc[msk_5, 'eight_trigrams'] = 5
    df.loc[msk_6, 'eight_trigrams'] = 6
    df.loc[msk_7, 'eight_trigrams'] = 7   
    features.extend(['eight_trigrams'])

    return df, features

def remove_outliers(df, features, threshold = 1000):
    for feat in features:
        df.loc[df[feat] > threshold, feat] = threshold
        df.loc[df[feat] < -threshold, feat] = -threshold
    return df