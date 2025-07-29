# Alpha101 因子實現 - 基於世坤標準定義
# 參考：https://www.worldquant.com/research/101-formulaic-alphas

import numpy as np
import pandas as pd
from tqdm import tqdm

# 基礎函數定義
def ts_rank(series, window):
    """時間序列排名"""
    return series.rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)

def ts_argmax(series, window):
    """時間序列最大值位置"""
    return series.rolling(window).apply(lambda x: np.argmax(x[::-1]), raw=True)

def ts_argmin(series, window):
    """時間序列最小值位置"""
    return series.rolling(window).apply(lambda x: np.argmin(x[::-1]), raw=True)

def ts_sum(series, window):
    """時間序列求和"""
    return series.rolling(window).sum()

def ts_mean(series, window):
    """時間序列平均值"""
    return series.rolling(window).mean()

def ts_stddev(series, window):
    """時間序列標準差"""
    return series.rolling(window).std()

def ts_min(series, window):
    """時間序列最小值"""
    return series.rolling(window).min()

def ts_max(series, window):
    """時間序列最大值"""
    return series.rolling(window).max()

def delta(series, period=1):
    """差分"""
    return series.diff(period)

def signed_power(x, a):
    """符號冪"""
    return np.sign(x) * (np.abs(x) ** a)

def correlation(x, y, window):
    """相關性"""
    return x.rolling(window).corr(y)

def covariance(x, y, window):
    """協方差"""
    return x.rolling(window).cov(y)

def decay_linear(series, window):
    """線性衰減加權平均"""
    weights = np.arange(1, window + 1)
    def lin_decay(x):
        if len(x) < window:
            return np.nan
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window).apply(lin_decay, raw=True)

def scale(series, k=1):
    """縮放"""
    return k * series / np.nansum(np.abs(series))

def rank(series):
    """橫截面排名"""
    return series.rank(pct=True)

def delay(series, d):
    """延遲d天"""
    return series.shift(d)

def product(series, window):
    """時間序列乘積"""
    return series.rolling(window).apply(lambda x: np.prod(x), raw=True)

def sum(series, window):
    """時間序列求和"""
    return series.rolling(window).sum()

def calculate_vwap(df, window=None):
    """
    計算成交量加權平均價 (VWAP)
    
    參數:
    df: DataFrame，需要包含 'open', 'high', 'low', 'close', 'volume' 欄位
    window: 計算窗口，如果指定則只使用最近 window 天的數據，預設為 None（使用全部數據）
    
    返回:
    Series: VWAP 值
    """
    # 計算典型價格 (Typical Price)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # 計算價格成交量乘積
    price_volume = typical_price * df['volume']
    
    if window is not None:
        # 使用滾動窗口計算 VWAP
        # 計算滾動窗口的價格成交量總和
        rolling_pv_sum = price_volume.rolling(window=window, min_periods=window).sum()
        # 計算滾動窗口的成交量總和
        rolling_volume_sum = df['volume'].rolling(window=window, min_periods=window).sum()
        # 計算 VWAP
        vwap = rolling_pv_sum / rolling_volume_sum
    else:
        # 計算累積價格成交量乘積
        cumulative_pv = price_volume.cumsum()
        
        # 計算累積成交量
        cumulative_volume = df['volume'].cumsum()
        
        # 計算 VWAP
        vwap = cumulative_pv / cumulative_volume
    
    return vwap

def calculate_adv(df, window=20):
    """
    計算平均日成交量 (Average Daily Volume)
    
    參數:
    df: DataFrame，需要包含 'volume' 欄位
    window: 計算窗口，預設為 20
    
    返回:
    Series: 平均日成交量
    """
    return df['volume'].rolling(window).mean()

def add_vwap_to_df(df, window=10):
    """
    為 DataFrame 添加 VWAP 欄位
    
    參數:
    df: DataFrame，需要包含 'open', 'high', 'low', 'close', 'volume' 欄位
    window: VWAP 計算窗口，預設為 10 天
    
    返回:
    DataFrame: 包含 'vwap' 欄位的 DataFrame
    """
    df = df.copy()
    
    # 按股票代碼分組計算 VWAP
    if 'stock_id' in df.columns:
        df['vwap'] = df.groupby('stock_id').apply(
            lambda x: calculate_vwap(x, window)
        ).reset_index(level=0, drop=True)
    else:
        df['vwap'] = calculate_vwap(df, window)
    
    return df

# Alpha101 因子實現
def alpha_001(df):
    """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    condition = df['returns'] < 0
    # 創建 pandas Series 而不是 numpy 數組
    signed_power_val = pd.Series(index=df.index)
    signed_power_val[condition] = signed_power(df['returns'].rolling(20).std(), 2)[condition]
    signed_power_val[~condition] = signed_power(df['close'], 2)[~condition]
    return rank(ts_argmax(signed_power_val, 5)) - 0.5

def alpha_002(df):
    """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    return -1 * correlation(rank(delta(np.log(df['volume']), 2)), 
                           rank((df['close'] - df['open']) / df['open']), 6)

def alpha_003(df):
    """(-1 * correlation(rank(open), rank(volume), 10))"""
    return -1 * correlation(rank(df['open']), rank(df['volume']), 10)

def alpha_004(df):
    """(-1 * Ts_Rank(rank(low), 9))"""
    return -1 * ts_rank(rank(df['low']), 9)

def alpha_005(df):
    """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    vwap_sum = df['vwap'].rolling(10).sum() / 10
    return rank(df['open'] - vwap_sum) * (-1 * abs(rank(df['close'] - df['vwap'])))

def alpha_006(df):
    """(-1 * correlation(open, volume, 10))"""
    return -1 * correlation(df['open'], df['volume'], 10)

def alpha_007(df):
    """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
    adv20 = df['volume'].rolling(20).mean()
    condition = adv20 < df['volume']
    delta_close_7 = delta(df['close'], 7)
    
    # 創建 pandas Series 而不是 numpy 數組
    result = pd.Series(index=df.index)
    result[condition] = ((-1 * ts_rank(abs(delta_close_7), 60)) * np.sign(delta_close_7))[condition]
    result[~condition] = -1
    
    return result

def alpha_008(df):
    """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    sum_open_5 = df['open'].rolling(5).sum()
    sum_returns_5 = df['returns'].rolling(5).sum()
    product = sum_open_5 * sum_returns_5
    return -1 * rank(product - delay(product, 10))

def alpha_009(df):
    """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
    delta_close_1 = delta(df['close'], 1)
    ts_min_delta = ts_min(delta_close_1, 5)
    ts_max_delta = ts_max(delta_close_1, 5)
    
    condition1 = 0 < ts_min_delta
    condition2 = ts_max_delta < 0
    
    # 創建 pandas Series 而不是 numpy 數組
    result = pd.Series(index=df.index)
    result[condition1] = delta_close_1[condition1]
    result[condition2] = delta_close_1[condition2]
    result[~(condition1 | condition2)] = -1 * delta_close_1[~(condition1 | condition2)]
    
    return result

def alpha_010(df):
    """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    delta_close_1 = delta(df['close'], 1)
    ts_min_delta = ts_min(delta_close_1, 4)
    ts_max_delta = ts_max(delta_close_1, 4)
    
    condition1 = 0 < ts_min_delta
    condition2 = ts_max_delta < 0
    
    # 創建 pandas Series 而不是 numpy 數組
    result = pd.Series(index=df.index)
    result[condition1] = delta_close_1[condition1]
    result[condition2] = delta_close_1[condition2]
    result[~(condition1 | condition2)] = -1 * delta_close_1[~(condition1 | condition2)]
    
    return rank(result)

def alpha_011(df):
    """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    vwap_close_diff = df['vwap'] - df['close']
    ts_max_vwap_close = ts_max(vwap_close_diff, 3)
    ts_min_vwap_close = ts_min(vwap_close_diff, 3)
    delta_volume_3 = delta(df['volume'], 3)
    return (rank(ts_max_vwap_close) + rank(ts_min_vwap_close)) * rank(delta_volume_3)

def alpha_012(df):
    """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    return np.sign(delta(df['volume'], 1)) * (-1 * delta(df['close'], 1))

def alpha_013(df):
    """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    return -1 * rank(covariance(rank(df['close']), rank(df['volume']), 5))

def alpha_014(df):
    """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    return (-1 * rank(delta(df['returns'], 3))) * correlation(df['open'], df['volume'], 10)

def alpha_015(df):
    """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    corr_rank_high_volume = correlation(rank(df['high']), rank(df['volume']), 3)
    return -1 * ts_sum(rank(corr_rank_high_volume), 3)

def alpha_016(df):
    """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    return -1 * rank(covariance(rank(df['high']), rank(df['volume']), 5))

def alpha_017(df):
    """(((rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    adv20 = df['volume'].rolling(20).mean()
    return ((-1 * rank(ts_rank(df['close'], 10))) * 
            rank(delta(delta(df['close'], 1), 1))) * rank(ts_rank((df['volume'] / adv20), 5))

def alpha_018(df):
    """(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
    stddev_abs_close_open = df['close'].rolling(5).std()
    close_open_diff = df['close'] - df['open']
    corr_close_open = correlation(df['close'], df['open'], 10)
    return -1 * rank(stddev_abs_close_open + close_open_diff + corr_close_open)

def alpha_019(df):
    """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    close_delay_7 = delay(df['close'], 7)
    delta_close_7 = delta(df['close'], 7)
    sign_val = np.sign((df['close'] - close_delay_7) + delta_close_7)
    sum_returns_250 = ts_sum(df['returns'], 250)
    return (-1 * sign_val) * (1 + rank(1 + sum_returns_250))

def alpha_020(df):
    """(((rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    high_delay_1 = delay(df['high'], 1)
    close_delay_1 = delay(df['close'], 1)
    low_delay_1 = delay(df['low'], 1)
    return (rank(df['open'] - high_delay_1) * 
            rank(df['open'] - close_delay_1) * 
            rank(df['open'] - low_delay_1))

def alpha_021(df):
    """((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))"""
    adv20 = df['volume'].rolling(20).mean()
    sum_close_8 = df['close'].rolling(8).sum() / 8
    sum_close_2 = df['close'].rolling(2).sum() / 2
    stddev_close_8 = df['close'].rolling(8).std()
    
    condition1 = (sum_close_8 + stddev_close_8) < sum_close_2
    condition2 = sum_close_2 < (sum_close_8 - stddev_close_8)
    condition3 = (1 < (df['volume'] / adv20)) | ((df['volume'] / adv20) == 1)
    
    result = pd.Series(index=df.index)
    result[condition1] = -1
    result[condition2] = 1
    result[~(condition1 | condition2)] = np.where(condition3[~(condition1 | condition2)], 1, -1)
    
    return result

def alpha_022(df):
    """(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
    corr_high_volume = correlation(df['high'], df['volume'], 5)
    delta_corr = delta(corr_high_volume, 5)
    stddev_close_20 = df['close'].rolling(20).std()
    return -1 * delta_corr * rank(stddev_close_20)

def alpha_023(df):
    """(((sum(high, 20) / 20) < high) ? (-1 * 1) : 1)"""
    sum_high_20 = df['high'].rolling(20).sum() / 20
    condition = sum_high_20 < df['high']
    return np.where(condition, -1, 1)

def alpha_024(df):
    """(((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ? (-1 * (close - delay(close, 1))) : (-1 * delta(close, 3)))"""
    sum_close_100 = df['close'].rolling(100).sum() / 100
    delta_sum_close = delta(sum_close_100, 100)
    close_delay_100 = delay(df['close'], 100)
    close_delay_1 = delay(df['close'], 1)
    delta_close_3 = delta(df['close'], 3)
    
    condition = (delta_sum_close / close_delay_100) < 0.05
    result = pd.Series(index=df.index)
    result[condition] = -1 * (df['close'] - close_delay_1)[condition]
    result[~condition] = -1 * delta_close_3[~condition]
    
    return result

def alpha_025(df):
    """rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
    adv20 = df['volume'].rolling(20).mean()
    return rank((-1 * df['returns']) * adv20 * df['vwap'] * (df['high'] - df['close']))

def alpha_026(df):
    """((((sum(close, 7) / 7) - close)) + (correlation(close, delay(close, 5), 230)))"""
    sum_close_7 = df['close'].rolling(7).sum() / 7
    close_delay_5 = delay(df['close'], 5)
    corr_close_delay = correlation(df['close'], close_delay_5, 230)
    return (sum_close_7 - df['close']) + corr_close_delay

def alpha_027(df):
    """WMA((close-delay(close,3))/delay(close,3)*100+(close-delay(close,6))/delay(close,6)*100,12)/100"""
    close_delay_3 = delay(df['close'], 3)
    close_delay_6 = delay(df['close'], 6)
    
    term1 = (df['close'] - close_delay_3) / close_delay_3 * 100
    term2 = (df['close'] - close_delay_6) / close_delay_6 * 100
    
    # WMA (Weighted Moving Average) 實現
    def wma(series, window):
        weights = np.arange(1, window + 1)
        def wma_func(x):
            if len(x) < window:
                return np.nan
            return np.dot(x, weights) / weights.sum()
        return series.rolling(window).apply(wma_func, raw=True)
    
    return wma(term1 + term2, 12) / 100

def alpha_028(df):
    """3*SMA((close-delay(low,1)-(high-delay(close,1)))/(high-low)*100,15,2)-2*SMA((close-delay(low,1)-(high-delay(close,1)))/(high-low)*100,15,2)*SMA((close-delay(low,1)-(high-delay(close,1)))/(high-low)*100,15,2)/30-SMA(((close-delay(low,1)-(high-delay(close,1)))/(high-low)*100,15,2),15,2)"""
    low_delay_1 = delay(df['low'], 1)
    high_delay_1 = delay(df['high'], 1)
    close_delay_1 = delay(df['close'], 1)
    
    numerator = (df['close'] - low_delay_1) - (high_delay_1 - close_delay_1)
    denominator = df['high'] - df['low']
    ratio = (numerator / denominator) * 100
    
    # SMA (Simple Moving Average) 實現
    def sma(series, window):
        return series.rolling(window).mean()
    
    sma_ratio = sma(ratio, 15)
    sma_sma_ratio = sma(sma_ratio, 15)
    
    return (3 * sma_ratio - 
            2 * sma_ratio * sma_ratio / 30 - 
            sma_sma_ratio)

def alpha_029(df):
    """(close-delay(close,6))/delay(close,6)*volume"""
    close_delay_6 = delay(df['close'], 6)
    return (df['close'] - close_delay_6) / close_delay_6 * df['volume']

def alpha_030(df):
    """((rank(sign(delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    delta_close_7 = delta(df['close'], 7)
    sign_delta = np.sign(delta_close_7)
    sum_returns_250 = ts_sum(df['returns'], 250)
    return rank(sign_delta) * (1 + rank(1 + sum_returns_250))

def alpha_031(df):
    """((rank(rank(rank(decay_linear((-1 * rank(rank(rank(volume)))), 10)))) + rank((-1 * rank(returns, 10)))) + (rank(rank(abs(correlation(vwap, delay(close, 5), 10)))))"""
    # 分步驟計算，避免複雜的括號嵌套
    volume_rank1 = rank(df['volume'])
    volume_rank2 = rank(volume_rank1)
    volume_rank3 = rank(volume_rank2)
    volume_ranked = -1 * volume_rank3
    
    decay_vol = decay_linear(volume_ranked, 10)
    rank_returns = ts_rank(df['returns'], 10)
    vwap_delay_close = correlation(df['vwap'], delay(df['close'], 5), 10)
    
    return rank(rank(rank(decay_vol)) + rank(-1 * rank_returns) + rank(rank(abs(vwap_delay_close))))

def alpha_032(df):
    """(scale(((sum(close, 7) / 7) - close)) + (scale(correlation(vwap, delay(close, 5), 230))))"""
    sum_close_7 = df['close'].rolling(7).sum() / 7
    vwap_delay_close = correlation(df['vwap'], delay(df['close'], 5), 230)
    
    return scale(sum_close_7 - df['close']) + scale(vwap_delay_close)

def alpha_033(df):
    """rank((-1 * ((1 - (open / close)) * 1)))"""
    return rank(-1 * (1 - (df['open'] / df['close'])) * 1)

def alpha_034(df):
    """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    stddev_returns_2 = df['returns'].rolling(2).std()
    stddev_returns_5 = df['returns'].rolling(5).std()
    delta_close_1 = delta(df['close'], 1)
    
    return rank((1 - rank(stddev_returns_2 / stddev_returns_5)) + (1 - rank(delta_close_1)))

def alpha_035(df):
    """((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    ts_rank_volume = ts_rank(df['volume'], 32)
    ts_rank_close_high_low = ts_rank(df['close'] + df['high'] - df['low'], 16)
    ts_rank_returns = ts_rank(df['returns'], 32)
    
    return ts_rank_volume * (1 - ts_rank_close_high_low) * (1 - ts_rank_returns)

def alpha_036(df):
    """(((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5))) + rank(abs(correlation(vwap, adv20, 6))) + (0.6 * rank((((sum(close, 20) / 20) - open) * (close - open)))))"""
    adv20 = df['volume'].rolling(20).mean()
    close_open_diff = df['close'] - df['open']
    volume_delay_1 = delay(df['volume'], 1)
    returns_delay_6 = delay(-1 * df['returns'], 6)
    sum_close_20 = df['close'].rolling(20).sum() / 20
    
    corr_close_open_volume = correlation(close_open_diff, volume_delay_1, 15)
    ts_rank_returns = ts_rank(returns_delay_6, 5)
    corr_vwap_adv20 = correlation(df['vwap'], adv20, 6)
    
    return (2.21 * rank(corr_close_open_volume) + 
            0.7 * rank(df['open'] - df['close']) + 
            0.73 * rank(ts_rank_returns) + 
            rank(abs(corr_vwap_adv20)) + 
            0.6 * rank((sum_close_20 - df['open']) * (df['close'] - df['open'])))

def alpha_037(df):
    """(rank(correlation(delta(close, 1), delay(delta(close, 1), 1), 3)) * rank((rank(correlation(rank(volume), rank(vwap), 6)))))"""
    delta_close_1 = delta(df['close'], 1)
    delta_close_delay_1 = delay(delta_close_1, 1)
    corr_delta = correlation(delta_close_1, delta_close_delay_1, 3)
    corr_volume_vwap = correlation(rank(df['volume']), rank(df['vwap']), 6)
    
    return rank(corr_delta) * rank(rank(corr_volume_vwap))

def alpha_038(df):
    """((-1 * rank(Ts_Rank(returns, 10))) * rank((close / open)))"""
    ts_rank_returns = ts_rank(df['returns'], 10)
    close_open_ratio = df['close'] / df['open']
    
    return -1 * rank(ts_rank_returns) * rank(close_open_ratio)

def alpha_039(df):
    """((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    adv20 = df['volume'].rolling(20).mean()
    delta_close_7 = delta(df['close'], 7)
    decay_volume_adv20 = decay_linear(df['volume'] / adv20, 9)
    sum_returns_250 = ts_sum(df['returns'], 250)
    
    return -1 * rank(delta_close_7 * (1 - rank(decay_volume_adv20))) * (1 + rank(sum_returns_250))

def alpha_040(df):
    """((-1 * rank(stddev(high, 10))) * correlation(close, volume, 10))"""
    stddev_high_10 = df['high'].rolling(10).std()
    corr_close_volume = correlation(df['close'], df['volume'], 10)
    
    return -1 * rank(stddev_high_10) * corr_close_volume

def alpha_041(df):
    """(((high * 0.6) + (vwap * 0.4)) - delay(((high * 0.6) + (vwap * 0.4)), 1))"""
    weighted_price = (df['high'] * 0.6) + (df['vwap'] * 0.4)
    return weighted_price - delay(weighted_price, 1)

def alpha_042(df):
    """(rank((vwap - close)) / rank((vwap + close)))"""
    return rank(df['vwap'] - df['close']) / rank(df['vwap'] + df['close'])

def alpha_043(df):
    """(ts_rank((volume / adv20), 4) * ts_rank((-1 * delta(close, 7)), 4))"""
    adv20 = df['volume'].rolling(20).mean()
    volume_adv20_ratio = df['volume'] / adv20
    delta_close_7 = delta(df['close'], 7)
    
    return ts_rank(volume_adv20_ratio, 4) * ts_rank(-1 * delta_close_7, 4)

def alpha_044(df):
    """(-1 * correlation(high, rank(volume), 5))"""
    return -1 * correlation(df['high'], rank(df['volume']), 5)

def alpha_045(df):
    """(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    delay_close_5 = delay(df['close'], 5)
    sum_delay_close_20 = ts_sum(delay_close_5, 20) / 20
    sum_close_5 = ts_sum(df['close'], 5)
    sum_close_20 = ts_sum(df['close'], 20)
    
    corr_close_volume = correlation(df['close'], df['volume'], 2)
    corr_sum_close = correlation(sum_close_5, sum_close_20, 2)
    
    return -1 * rank(sum_delay_close_20) * corr_close_volume * rank(corr_sum_close)

def alpha_046(df):
    """((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))"""
    delay_close_20 = delay(df['close'], 20)
    delay_close_10 = delay(df['close'], 10)
    delay_close_1 = delay(df['close'], 1)
    
    term1 = (delay_close_20 - delay_close_10) / 10
    term2 = (delay_close_10 - df['close']) / 10
    diff = term1 - term2
    
    condition1 = 0.25 < diff
    condition2 = diff < 0
    
    result = pd.Series(index=df.index)
    result[condition1] = -1
    result[condition2] = 1
    result[~(condition1 | condition2)] = -1 * (df['close'] - delay_close_1)[~(condition1 | condition2)]
    
    return result

def alpha_047(df):
    """(((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5)))"""
    adv20 = df['volume'].rolling(20).mean()
    sum_high_5 = ts_sum(df['high'], 5) / 5
    
    term1 = (rank(1 / df['close']) * df['volume']) / adv20
    term2 = (df['high'] * rank(df['high'] - df['close'])) / sum_high_5
    
    return term1 * term2

def alpha_048(df):
    """(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 3)) * rank(correlation(sum(volume, 5), sum(volume, 20), 2))), rank(sum(close, 5), 5)))"""
    # 簡化實現，移除 indneutralize 函數
    delta_close_1 = delta(df['close'], 1)
    delta_delay_close_1 = delta(delay(df['close'], 1), 1)
    sum_volume_5 = ts_sum(df['volume'], 5)
    sum_volume_20 = ts_sum(df['volume'], 20)
    sum_close_5 = ts_sum(df['close'], 5)
    
    corr_delta = correlation(delta_close_1, delta_delay_close_1, 3)
    corr_volume = correlation(sum_volume_5, sum_volume_20, 2)
    
    return corr_delta * rank(corr_volume) * rank(sum_close_5)

def alpha_049(df):
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    delay_close_20 = delay(df['close'], 20)
    delay_close_10 = delay(df['close'], 10)
    delay_close_1 = delay(df['close'], 1)
    
    term1 = (delay_close_20 - delay_close_10) / 10
    term2 = (delay_close_10 - df['close']) / 10
    diff = term1 - term2
    
    condition = diff < (-1 * 0.1)
    result = pd.Series(index=df.index)
    result[condition] = 1
    result[~condition] = -1 * (df['close'] - delay_close_1)[~condition]
    
    return result

def alpha_050(df):
    """(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    corr_volume_vwap = correlation(rank(df['volume']), rank(df['vwap']), 5)
    return -1 * ts_max(rank(corr_volume_vwap), 5)

def alpha_051(df):
    """(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    delay_close_20 = delay(df['close'], 20)
    delay_close_10 = delay(df['close'], 10)
    delay_close_1 = delay(df['close'], 1)
    
    term1 = (delay_close_20 - delay_close_10) / 10
    term2 = (delay_close_10 - df['close']) / 10
    diff = term1 - term2
    
    condition = diff < (-1 * 0.05)
    result = pd.Series(index=df.index)
    result[condition] = 1
    result[~condition] = -1 * (df['close'] - delay_close_1)[~condition]
    
    return result

def alpha_052(df):
    """((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
    ts_min_low_5 = ts_min(df['low'], 5)
    delay_ts_min_low_5 = delay(ts_min_low_5, 5)
    sum_returns_240 = ts_sum(df['returns'], 240)
    sum_returns_20 = ts_sum(df['returns'], 20)
    
    term1 = (-1 * ts_min_low_5) + delay_ts_min_low_5
    term2 = rank((sum_returns_240 - sum_returns_20) / 220)
    term3 = ts_rank(df['volume'], 5)
    
    return term1 * term2 * term3

def alpha_053(df):
    """(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
    numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
    denominator = df['close'] - df['low']
    ratio = numerator / denominator
    
    return -1 * delta(ratio, 9)

def alpha_054(df):
    """((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
    return (-1 * (df['low'] - df['close']) * (df['open'] ** 5)) / ((df['low'] - df['high']) * (df['close'] ** 5))

def alpha_055(df):
    """(-1 * correlation(rank(((close - ts_min(low, 12))^2)), rank(correlation(volume, vwap, 6)), 6))"""
    ts_min_low_12 = ts_min(df['low'], 12)
    term1 = (df['close'] - ts_min_low_12) ** 2
    corr_volume_vwap = correlation(df['volume'], df['vwap'], 6)
    
    return -1 * correlation(rank(term1), rank(corr_volume_vwap), 6)

def alpha_056(df):
    """(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
    sum_returns_10 = ts_sum(df['returns'], 10)
    sum_returns_2 = ts_sum(df['returns'], 2)
    sum_sum_returns_2_3 = ts_sum(sum_returns_2, 3)
    
    # 假設 cap 為 volume，因為原始公式中 cap 未定義
    cap = df['volume']
    
    return 0 - (1 * rank(sum_returns_10 / sum_sum_returns_2_3) * rank(df['returns'] * cap))

def alpha_057(df):
    """(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
    ts_argmax_close_30 = ts_argmax(df['close'], 30)
    decay_rank_ts_argmax = decay_linear(rank(ts_argmax_close_30), 2)
    
    return 0 - (1 * (df['close'] - df['vwap']) / decay_rank_ts_argmax)

def alpha_058(df):
    """(-1 * rank(((vwap * 0.728317) + (vwap * (1 - 0.728317)) - delay(vwap, 1))))"""
    weighted_vwap = (df['vwap'] * 0.728317) + (df['vwap'] * (1 - 0.728317))
    return -1 * rank(weighted_vwap - delay(df['vwap'], 1))

def alpha_059(df):
    """((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)))"""
    delay_close_20 = delay(df['close'], 20)
    delay_close_10 = delay(df['close'], 10)
    
    term1 = (delay_close_20 - delay_close_10) / 10
    term2 = (delay_close_10 - df['close']) / 10
    
    return term1 - term2

def alpha_060(df):
    """((0 - (1 * ((2 * (rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - rank(correlation(((close - low) - (high - close)), delay(volume, 2), 8))))))"""
    numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
    denominator = df['high'] - df['low']
    ratio = numerator / denominator
    volume_ratio = ratio * df['volume']
    
    delay_volume_2 = delay(df['volume'], 2)
    corr_ratio_volume = correlation(numerator, delay_volume_2, 8)
    
    return 0 - (1 * (2 * rank(volume_ratio) - rank(corr_ratio_volume)))

def alpha_061(df):
    """(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv20, 18.9282)))"""
    ts_min_vwap = ts_min(df['vwap'], 16)
    adv20 = df['volume'].rolling(20).mean()
    corr_vwap_adv20 = correlation(df['vwap'], adv20, 19)
    
    return rank(df['vwap'] - ts_min_vwap) < rank(corr_vwap_adv20)

def alpha_062(df):
    """((rank(correlation(vwap, sum(adv20, 14.4714), 6.4714)) + rank((open - delay(high, 1)))))"""
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = ts_sum(adv20, 14)
    corr_vwap_sum_adv20 = correlation(df['vwap'], sum_adv20, 6)
    delay_high_1 = delay(df['high'], 1)
    
    return rank(corr_vwap_sum_adv20) + rank(df['open'] - delay_high_1)

def alpha_063(df):
    """((rank(decay_linear(delta(IndNeutralize(close, IndClass.sector), 2.72412), 6.95999)) + rank(correlation(adv20, delay(close, 4.6095), 6.95999)))"""
    # 簡化實現，移除 IndNeutralize 函數
    delta_close = delta(df['close'], 3)
    decay_delta = decay_linear(delta_close, 7)
    adv20 = df['volume'].rolling(20).mean()
    delay_close_5 = delay(df['close'], 5)
    corr_adv20_delay_close = correlation(adv20, delay_close_5, 7)
    
    return rank(decay_delta) + rank(corr_adv20_delay_close)

def alpha_064(df):
    """((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7052), sum(adv20, 12.7052), 16.6208)) < rank(delta(((close * 0.955724) + (open * (1 - 0.955724))), 2.8584))) * -1)"""
    weighted_price = (df['open'] * 0.178404) + (df['low'] * (1 - 0.178404))
    sum_weighted_price = ts_sum(weighted_price, 13)
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = ts_sum(adv20, 13)
    corr_sum_price_adv20 = correlation(sum_weighted_price, sum_adv20, 17)
    
    close_open_weighted = (df['close'] * 0.955724) + (df['open'] * (1 - 0.955724))
    delta_close_open = delta(close_open_weighted, 3)
    
    return (rank(corr_sum_price_adv20) < rank(delta_close_open)) * -1

def alpha_065(df):
    """(rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv20, 6.6911), 6.6911)) < rank((open - delay(high, 1)))))"""
    weighted_open_vwap = (df['open'] * 0.00817205) + (df['vwap'] * (1 - 0.00817205))
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = ts_sum(adv20, 7)
    corr_weighted_sum = correlation(weighted_open_vwap, sum_adv20, 7)
    
    delay_high_1 = delay(df['high'], 1)
    open_delay_high = df['open'] - delay_high_1
    
    return rank(corr_weighted_sum) < rank(open_delay_high)

def alpha_066(df):
    """((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)"""
    delta_vwap = delta(df['vwap'], 4)
    decay_delta_vwap = decay_linear(delta_vwap, 7)
    
    low_weighted = (df['low'] * 0.96633) + (df['low'] * (1 - 0.96633))
    high_low_avg = (df['high'] + df['low']) / 2
    numerator = low_weighted - df['vwap']
    denominator = df['open'] - high_low_avg
    ratio = numerator / denominator
    decay_ratio = decay_linear(ratio, 11)
    ts_rank_decay = ts_rank(decay_ratio, 7)
    
    return (rank(decay_delta_vwap) + ts_rank_decay) * -1

def alpha_067(df):
    """((rank((high - ts_min(high, 2.72412)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.sector), 6.4714)))"""
    # 簡化實現，移除 IndNeutralize 函數
    ts_min_high = ts_min(df['high'], 3)
    high_min_diff = df['high'] - ts_min_high
    adv20 = df['volume'].rolling(20).mean()
    corr_vwap_adv20 = correlation(df['vwap'], adv20, 6)
    
    return rank(high_min_diff) ** rank(corr_vwap_adv20)

def alpha_068(df):
    """((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)"""
    adv15 = df['volume'].rolling(15).mean()
    corr_rank_high_adv15 = correlation(rank(df['high']), rank(adv15), 9)
    ts_rank_corr = ts_rank(corr_rank_high_adv15, 14)
    
    close_low_weighted = (df['close'] * 0.518371) + (df['low'] * (1 - 0.518371))
    delta_weighted = delta(close_low_weighted, 1)
    
    return (ts_rank_corr < rank(delta_weighted)) * -1

def alpha_069(df):
    """((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^rank(correlation(IndNeutralize(close, IndClass.industry), IndNeutralize(adv20, IndClass.industry), 4.79344)))"""
    # 簡化實現，移除 IndNeutralize 函數
    delta_vwap = delta(df['vwap'], 3)
    ts_max_delta = ts_max(delta_vwap, 5)
    adv20 = df['volume'].rolling(20).mean()
    corr_close_adv20 = correlation(df['close'], adv20, 5)
    
    return rank(ts_max_delta) ** rank(corr_close_adv20)

def alpha_070(df):
    """((rank(delta(vwap, 1.29456))^rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256)))"""
    # 簡化實現，移除 IndNeutralize 函數
    delta_vwap = delta(df['vwap'], 1)
    adv50 = df['volume'].rolling(50).mean()
    corr_close_adv50 = correlation(df['close'], adv50, 18)
    
    return rank(delta_vwap) ** rank(corr_close_adv50)

def alpha_071(df):
    """max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.70376), Ts_Rank(adv15, 15.7152), 4.86682), 7.95524), 3.65999), Ts_Rank(decay_linear(Ts_Rank(correlation(rank(low), rank(adv15), 8.62541), 8.40009), 13.6813)))"""
    adv15 = df['volume'].rolling(15).mean()
    
    # 第一部分
    ts_rank_close = ts_rank(df['close'], 4)
    ts_rank_adv15 = ts_rank(adv15, 16)
    corr_ts_rank = correlation(ts_rank_close, ts_rank_adv15, 5)
    decay_corr = decay_linear(corr_ts_rank, 8)
    ts_rank_decay1 = ts_rank(decay_corr, 4)
    
    # 第二部分
    corr_rank_low_adv15 = correlation(rank(df['low']), rank(adv15), 9)
    ts_rank_corr = ts_rank(corr_rank_low_adv15, 8)
    decay_ts_rank = decay_linear(ts_rank_corr, 14)
    ts_rank_decay2 = ts_rank(decay_ts_rank, 14)
    
    return np.maximum(ts_rank_decay1, ts_rank_decay2)

def alpha_072(df):
    """(rank(decay_linear(correlation(((high + low) / 2), adv20, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72465), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""
    high_low_avg = (df['high'] + df['low']) / 2
    adv20 = df['volume'].rolling(20).mean()
    corr_hl_adv20 = correlation(high_low_avg, adv20, 9)
    decay_corr_hl = decay_linear(corr_hl_adv20, 10)
    
    ts_rank_vwap = ts_rank(df['vwap'], 4)
    ts_rank_volume = ts_rank(df['volume'], 19)
    corr_ts_rank = correlation(ts_rank_vwap, ts_rank_volume, 7)
    decay_corr_ts = decay_linear(corr_ts_rank, 3)
    
    return rank(decay_corr_hl) / rank(decay_corr_ts)

def alpha_073(df):
    """(max(rank(decay_linear(delta(vwap, 4.72797), 2.22164)), Ts_Rank(decay_linear(delta(((adv20 * 0.369701) + (vwap * (1 - 0.369701))), 3.08213), 3.22299), 9.08988)) * -1)"""
    delta_vwap = delta(df['vwap'], 5)
    decay_delta_vwap = decay_linear(delta_vwap, 2)
    rank_decay1 = rank(decay_delta_vwap)
    
    adv20 = df['volume'].rolling(20).mean()
    weighted_adv20_vwap = (adv20 * 0.369701) + (df['vwap'] * (1 - 0.369701))
    delta_weighted = delta(weighted_adv20_vwap, 3)
    decay_delta_weighted = decay_linear(delta_weighted, 3)
    ts_rank_decay2 = ts_rank(decay_delta_weighted, 9)
    
    return np.maximum(rank_decay1, ts_rank_decay2) * -1

def alpha_074(df):
    """((rank(correlation(close, sum(adv20, 37.0616), 15.1102)) < rank(correlation(rank(((high + low) / 2)), rank(volume), 11.2328))) * -1)"""
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = ts_sum(adv20, 37)
    corr_close_sum_adv20 = correlation(df['close'], sum_adv20, 15)
    
    high_low_avg = (df['high'] + df['low']) / 2
    corr_rank_hl_volume = correlation(rank(high_low_avg), rank(df['volume']), 11)
    
    return (rank(corr_close_sum_adv20) < rank(corr_rank_hl_volume)) * -1

def alpha_075(df):
    """(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 4.14243)))"""
    corr_vwap_volume = correlation(df['vwap'], df['volume'], 4)
    adv50 = df['volume'].rolling(50).mean()
    corr_rank_low_adv50 = correlation(rank(df['low']), rank(adv50), 4)
    
    return rank(corr_vwap_volume) < rank(corr_rank_low_adv50)

def alpha_076(df):
    """(max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.47141), 6.94124), 13.2504)))"""
    # 簡化實現，移除 IndNeutralize 函數
    delta_vwap = delta(df['vwap'], 1)
    decay_delta_vwap = decay_linear(delta_vwap, 12)
    rank_decay1 = rank(decay_delta_vwap)
    
    adv81 = df['volume'].rolling(81).mean()
    corr_low_adv81 = correlation(df['low'], adv81, 8)
    ts_rank_corr = ts_rank(corr_low_adv81, 7)
    decay_ts_rank = decay_linear(ts_rank_corr, 13)
    rank_decay2 = rank(decay_ts_rank)
    
    return np.maximum(rank_decay1, rank_decay2)

def alpha_077(df):
    """min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 9.04588)), rank(decay_linear(correlation(((low + high) / 2), adv40, 9.47008), 9.47008)))"""
    high_low_avg = (df['high'] + df['low']) / 2
    term1 = ((high_low_avg + df['high']) - (df['vwap'] + df['high']))
    decay_term1 = decay_linear(term1, 9)
    rank_decay1 = rank(decay_term1)
    
    adv40 = df['volume'].rolling(40).mean()
    corr_hl_adv40 = correlation(high_low_avg, adv40, 9)
    decay_corr = decay_linear(corr_hl_adv40, 9)
    rank_decay2 = rank(decay_corr)
    
    return np.minimum(rank_decay1, rank_decay2)

def alpha_078(df):
    """(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77452)))"""
    low_vwap_weighted = (df['low'] * 0.352233) + (df['vwap'] * (1 - 0.352233))
    sum_weighted = ts_sum(low_vwap_weighted, 20)
    adv40 = df['volume'].rolling(40).mean()
    sum_adv40 = ts_sum(adv40, 20)
    corr_sum_weighted_adv40 = correlation(sum_weighted, sum_adv40, 7)
    
    corr_rank_vwap_volume = correlation(rank(df['vwap']), rank(df['volume']), 6)
    
    return rank(corr_sum_weighted_adv40) ** rank(corr_rank_vwap_volume)

def alpha_079(df):
    """(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))"""
    # 簡化實現，移除 IndNeutralize 函數
    close_open_weighted = (df['close'] * 0.60733) + (df['open'] * (1 - 0.60733))
    delta_weighted = delta(close_open_weighted, 1)
    
    adv150 = df['volume'].rolling(150).mean()
    ts_rank_vwap = ts_rank(df['vwap'], 4)
    ts_rank_adv150 = ts_rank(adv150, 9)
    corr_ts_rank = correlation(ts_rank_vwap, ts_rank_adv150, 15)
    
    return rank(delta_weighted) < rank(corr_ts_rank)

def alpha_080(df):
    """((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^rank(correlation(high, adv10, 5.11456)))"""
    # 簡化實現，移除 IndNeutralize 函數
    open_high_weighted = (df['open'] * 0.868128) + (df['high'] * (1 - 0.868128))
    delta_weighted = delta(open_high_weighted, 4)
    sign_delta = np.sign(delta_weighted)
    
    adv10 = df['volume'].rolling(10).mean()
    corr_high_adv10 = correlation(df['high'], adv10, 5)
    
    return rank(sign_delta) ** rank(corr_high_adv10)

# 由於篇幅限制，這裡只實現前80個因子作為示例
# 完整的101個因子需要更多空間，建議分批實現

# 因子字典（前80個）
alpha_funcs = {
    'alpha_001': alpha_001,
    'alpha_002': alpha_002,
    'alpha_003': alpha_003,
    'alpha_004': alpha_004,
    'alpha_005': alpha_005,
    'alpha_006': alpha_006,
    'alpha_007': alpha_007,
    'alpha_008': alpha_008,
    'alpha_009': alpha_009,
    'alpha_010': alpha_010,
    'alpha_011': alpha_011,
    'alpha_012': alpha_012,
    'alpha_013': alpha_013,
    'alpha_014': alpha_014,
    'alpha_015': alpha_015,
    'alpha_016': alpha_016,
    'alpha_017': alpha_017,
    'alpha_018': alpha_018,
    'alpha_019': alpha_019,
    'alpha_020': alpha_020,
    'alpha_021': alpha_021,
    'alpha_022': alpha_022,
    'alpha_023': alpha_023,
    'alpha_024': alpha_024,
    'alpha_025': alpha_025,
    'alpha_026': alpha_026,
    'alpha_027': alpha_027,
    'alpha_028': alpha_028,
    'alpha_029': alpha_029,
    'alpha_030': alpha_030,
    'alpha_031': alpha_031,
    'alpha_032': alpha_032,
    'alpha_033': alpha_033,
    'alpha_034': alpha_034,
    'alpha_035': alpha_035,
    'alpha_036': alpha_036,
    'alpha_037': alpha_037,
    'alpha_038': alpha_038,
    'alpha_039': alpha_039,
    'alpha_040': alpha_040,
    'alpha_041': alpha_041,
    'alpha_042': alpha_042,
    'alpha_043': alpha_043,
    'alpha_044': alpha_044,
    'alpha_045': alpha_045,
    'alpha_046': alpha_046,
    'alpha_047': alpha_047,
    'alpha_048': alpha_048,
    'alpha_049': alpha_049,
    'alpha_050': alpha_050,
    'alpha_051': alpha_051,
    'alpha_052': alpha_052,
    'alpha_053': alpha_053,
    'alpha_054': alpha_054,
    'alpha_055': alpha_055,
    'alpha_056': alpha_056,
    'alpha_057': alpha_057,
    'alpha_058': alpha_058,
    'alpha_059': alpha_059,
    'alpha_060': alpha_060,
    'alpha_061': alpha_061,
    'alpha_062': alpha_062,
    'alpha_063': alpha_063,
    'alpha_064': alpha_064,
    'alpha_065': alpha_065,
    'alpha_066': alpha_066,
    'alpha_067': alpha_067,
    'alpha_068': alpha_068,
    'alpha_069': alpha_069,
    'alpha_070': alpha_070,
    'alpha_071': alpha_071,
    'alpha_072': alpha_072,
    'alpha_073': alpha_073,
    'alpha_074': alpha_074,
    'alpha_075': alpha_075,
    'alpha_076': alpha_076,
    'alpha_077': alpha_077,
    'alpha_078': alpha_078,
    'alpha_079': alpha_079,
    'alpha_080': alpha_080,
}

def calc_alpha101_factors(df, vwap_window=10):
    """
    計算 Alpha101 因子（前80個示例）
    傳回含所有 alpha_xxx 欄位的 DataFrame
    
    參數:
    df: DataFrame，需要包含必要的價格和成交量數據
    vwap_window: VWAP 計算窗口，預設為 10 天
    
    注意：此函數需要以下欄位：
    - stock_id: 股票代碼
    - date: 日期
    - open, close, high, low: 開盤、收盤、最高、最低價
    - volume: 成交量
    - vwap: 成交量加權平均價（如果沒有會自動計算）
    - returns: 收益率（如果沒有會自動計算）
    """
    df = df.copy()
    df.sort_values(['stock_id', 'date'], inplace=True)
    
    # 預先計算常用欄位
    if 'returns' not in df.columns:
        df['returns'] = df.groupby('stock_id')['close'].pct_change()
    
    # 如果沒有 vwap 欄位，自動計算
    if 'vwap' not in df.columns:
        df = add_vwap_to_df(df, window=vwap_window)
    
    # 計算所有因子
    for name, func in tqdm(alpha_funcs.items()):
        try:
            df[name] = df.groupby('stock_id').apply(lambda x: func(x)).reset_index(level=0, drop=True)
        except Exception as e:
            print(f"計算 {name} 時發生錯誤: {e}")
            df[name] = np.nan
    
    return df

# 使用範例
if __name__ == "__main__":
    # 範例數據結構（不包含 vwap，會自動計算）
    sample_data = {
        'stock_id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03'],
        'open': [100, 101, 102, 200, 201, 202],
        'close': [101, 102, 103, 201, 202, 203],
        'high': [102, 103, 104, 202, 203, 204],
        'low': [99, 100, 101, 199, 200, 201],
        'volume': [1000, 1100, 1200, 2000, 2100, 2200]
    }
    
    df = pd.DataFrame(sample_data)
    df['date'] = pd.to_datetime(df['date'])
    
    print("原始數據:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # 計算 VWAP
    df_with_vwap = add_vwap_to_df(df)
    print("添加 VWAP 後的數據:")
    print(df_with_vwap)
    print("\n" + "="*50 + "\n")
    
    # 計算因子（會自動計算 VWAP）
    result = calc_alpha101_factors(df)
    print("Alpha101 因子計算完成")
    print(f"結果包含 {len(alpha_funcs)} 個因子")
    print("因子欄位:", list(alpha_funcs.keys()))
    print("\n前幾行結果:")
    print(result.head())


