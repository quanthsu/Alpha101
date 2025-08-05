# Alpha101 因子實現 - 基於世坤標準定義
# 參考：https://www.worldquant.com/research/101-formulaic-alphas

import numpy as np
import pandas as pd
from tqdm import tqdm

# 基礎函數定義
def ts_rank(series, window):
    """時間序列排名"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    # 實際的分組會在調用時處理
    return series.rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False)

def ts_argmax(series, window):
    """時間序列最大值位置"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).apply(lambda x: np.argmax(x[::-1]), raw=True)

def ts_argmin(series, window):
    """時間序列最小值位置"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).apply(lambda x: np.argmin(x[::-1]), raw=True)

def ts_sum(series, window):
    """時間序列求和"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).sum()

def ts_mean(series, window):
    """時間序列平均值"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).mean()

def ts_stddev(series, window):
    """時間序列標準差"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).std()

def ts_min(series, window):
    """時間序列最小值"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).min()

def ts_max(series, window):
    """時間序列最大值"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).max()

def delta(series, period=1):
    """差分"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.diff(period)

def signed_power(x, a):
    """符號冪"""
    return np.sign(x) * (np.abs(x) ** a)

def correlation(x, y, window):
    """相關性"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return x.rolling(window).corr(y)

def covariance(x, y, window):
    """協方差"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return x.rolling(window).cov(y)

def decay_linear(series, window):
    """線性衰減加權平均"""
    weights = np.arange(1, window + 1)
    def lin_decay(x):
        if len(x) < window:
            return np.nan
        return np.dot(x, weights) / weights.sum()
    
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).apply(lin_decay, raw=True)

def scale(series, k=1):
    """縮放"""
    return k * series / np.nansum(np.abs(series))

def rank(series):
    """橫截面排名"""
    # 這個函數需要在全局 DataFrame 上調用，按日期分組進行橫截面排名
    # 注意：這個函數需要在包含所有股票數據的 DataFrame 上調用
    return series.rank(pct=True)

def cross_sectional_rank(df, series_or_column):
    """橫截面排名函數"""
    if isinstance(series_or_column, str):
        # 如果輸入是欄位名稱
        return df.groupby('date')[series_or_column].rank(pct=True)
    else:
        # 如果輸入是 Series 或 numpy array
        series = series_or_column
        
        # 檢查是否為 numpy array
        if hasattr(series, 'dtype') and hasattr(series, 'shape'):
            # 如果是 numpy array，轉換為 Series
            if df.index.is_unique:
                series = pd.Series(series, index=df.index)
            else:
                # 如果 index 不唯一，直接創建 Series 確保長度匹配
                series = pd.Series(series, index=df.index)
        elif hasattr(series, 'name') and series.name in df.columns:
            return df.groupby('date')[series.name].rank(pct=True)
        
        # 創建臨時欄位
        temp_col = f'temp_{id(series)}'
        df[temp_col] = series
        result = df.groupby('date')[temp_col].rank(pct=True)
        df.drop(columns=[temp_col], inplace=True)
        return result

def delay(series, d):
    """延遲d天"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.shift(d)

def product(series, window):
    """時間序列乘積"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
    return series.rolling(window).apply(lambda x: np.prod(x), raw=True)

def sum(series, window):
    """時間序列求和"""
    # 這個函數需要在包含 stock_id 的 DataFrame 上調用
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

def apply_ts_function_by_stock(df, column_or_series, func, window, **kwargs):
    """
    按股票分組應用時間序列函數（優化版本）
    
    參數:
    df: DataFrame，包含 stock_id 欄位
    column_or_series: 要計算的欄位名稱或 Series 或 numpy array
    func: 時間序列函數（如 ts_sum, ts_mean 等）
    window: 窗口大小
    **kwargs: 其他參數
    
    返回:
    Series: 按股票分組計算的結果
    """
    # 確保 DataFrame 有 stock_id 欄位
    if 'stock_id' not in df.columns:
        raise ValueError("DataFrame 必須包含 'stock_id' 欄位")
    
    # 準備要計算的 Series
    if isinstance(column_or_series, str):
        series_to_calc = df[column_or_series]
    else:
        # 如果是 Series 或 numpy array，確保索引匹配
        if hasattr(column_or_series, 'reindex'):
            # 如果是 pandas Series，檢查索引是否唯一
            if df.index.is_unique:
                series_to_calc = column_or_series.reindex(df.index)
            else:
                # 如果索引不唯一，直接使用原始數據並確保長度匹配
                if len(column_or_series) == len(df):
                    series_to_calc = pd.Series(column_or_series, index=df.index)
                else:
                    # 如果長度不匹配，使用前N個值
                    series_to_calc = pd.Series(column_or_series[:len(df)], index=df.index)
        else:
            # 如果是 numpy array，轉換為 Series
            if len(column_or_series) == len(df):
                series_to_calc = pd.Series(column_or_series, index=df.index)
            else:
                # 如果長度不匹配，使用前N個值
                series_to_calc = pd.Series(column_or_series[:len(df)], index=df.index)
    
    # 使用 groupby().rolling() 進行向量化計算
    if func == ts_sum:
        result = series_to_calc.groupby(df['stock_id']).rolling(window).sum().reset_index(0, drop=True)
    elif func == ts_mean:
        result = series_to_calc.groupby(df['stock_id']).rolling(window).mean().reset_index(0, drop=True)
    elif func == ts_stddev:
        result = series_to_calc.groupby(df['stock_id']).rolling(window).std().reset_index(0, drop=True)
    elif func == ts_min:
        result = series_to_calc.groupby(df['stock_id']).rolling(window).min().reset_index(0, drop=True)
    elif func == ts_max:
        result = series_to_calc.groupby(df['stock_id']).rolling(window).max().reset_index(0, drop=True)
    elif func == ts_rank:
        def rank_func(x):
            if len(x) < window:
                return np.nan
            return pd.Series(x).rank().iloc[-1] / len(x)
        result = series_to_calc.groupby(df['stock_id']).rolling(window).apply(rank_func, raw=False).reset_index(0, drop=True)
    elif func == ts_argmax:
        def argmax_func(x):
            if len(x) < window:
                return np.nan
            return np.argmax(x[::-1])
        result = series_to_calc.groupby(df['stock_id']).rolling(window).apply(argmax_func, raw=True).reset_index(0, drop=True)
    elif func == ts_argmin:
        def argmin_func(x):
            if len(x) < window:
                return np.nan
            return np.argmin(x[::-1])
        result = series_to_calc.groupby(df['stock_id']).rolling(window).apply(argmin_func, raw=True).reset_index(0, drop=True)
    elif func == delta:
        period = kwargs.get('period', 1)
        result = series_to_calc.groupby(df['stock_id']).diff(period)
    elif func == delay:
        d = kwargs.get('d', 1)
        result = series_to_calc.groupby(df['stock_id']).shift(d)
    elif func == product:
        def product_func(x):
            if len(x) < window:
                return np.nan
            return np.prod(x)
        result = series_to_calc.groupby(df['stock_id']).rolling(window).apply(product_func, raw=True).reset_index(0, drop=True)
    elif func == decay_linear:
        weights = np.arange(1, window + 1)
        def lin_decay(x):
            if len(x) < window:
                return np.nan
            return np.dot(x, weights) / weights.sum()
        result = series_to_calc.groupby(df['stock_id']).rolling(window).apply(lin_decay, raw=True).reset_index(0, drop=True)
    else:
        # 對於其他函數，使用通用方法
        result = series_to_calc.groupby(df['stock_id']).rolling(window).apply(func, raw=True).reset_index(0, drop=True)
    
    return result

def apply_correlation_by_stock(df, col1_or_series1, col2_or_series2, window):
    """
    按股票分組計算相關性（使用 date 索引優化版本）
    
    參數:
    df: DataFrame，包含 stock_id 和 date 欄位
    col1_or_series1, col2_or_series2: 要計算相關性的兩個欄位名稱或 Series 或 numpy array
    window: 窗口大小（以天為單位）
    
    返回:
    Series: 按股票分組並按日期索引計算的相關性
    """
    # 確保 DataFrame 有必要欄位
    if 'stock_id' not in df.columns:
        raise ValueError("DataFrame 必須包含 'stock_id' 欄位")
    if 'date' not in df.columns:
        raise ValueError("DataFrame 必須包含 'date' 欄位")
    
    # 創建 DataFrame 副本以避免修改原始數據
    work_df = df.copy()
    
    # 生成臨時欄位名稱
    temp_col1 = '_temp_corr_col1_'
    temp_col2 = '_temp_corr_col2_'
    
    # 準備兩個要計算相關性的 Series，並暫時加入 DataFrame
    if isinstance(col1_or_series1, str):
        work_df[temp_col1] = work_df[col1_or_series1]
    else:
        # 如果是 Series 或 numpy array，轉換並加入 DataFrame
        if hasattr(col1_or_series1, 'reindex'):
            # 如果是 pandas Series
            if work_df.index.is_unique:
                work_df[temp_col1] = col1_or_series1.reindex(work_df.index)
            else:
                if len(col1_or_series1) == len(work_df):
                    work_df[temp_col1] = pd.Series(col1_or_series1, index=work_df.index).values
                else:
                    work_df[temp_col1] = pd.Series(col1_or_series1[:len(work_df)], index=work_df.index).values
        else:
            # 如果是 numpy array
            if len(col1_or_series1) == len(work_df):
                work_df[temp_col1] = col1_or_series1
            else:
                work_df[temp_col1] = col1_or_series1[:len(work_df)]
    
    if isinstance(col2_or_series2, str):
        work_df[temp_col2] = work_df[col2_or_series2]
    else:
        # 如果是 Series 或 numpy array，轉換並加入 DataFrame
        if hasattr(col2_or_series2, 'reindex'):
            # 如果是 pandas Series
            if work_df.index.is_unique:
                work_df[temp_col2] = col2_or_series2.reindex(work_df.index)
            else:
                if len(col2_or_series2) == len(work_df):
                    work_df[temp_col2] = pd.Series(col2_or_series2, index=work_df.index).values
                else:
                    work_df[temp_col2] = pd.Series(col2_or_series2[:len(work_df)], index=work_df.index).values
        else:
            # 如果是 numpy array
            if len(col2_or_series2) == len(work_df):
                work_df[temp_col2] = col2_or_series2
            else:
                work_df[temp_col2] = col2_or_series2[:len(work_df)]
    
    # 確保 date 欄位為 datetime 格式並設置為索引，同時保持股票分組
    if not pd.api.types.is_datetime64_any_dtype(work_df['date']):
        work_df['date'] = pd.to_datetime(work_df['date'])
    
    # 按股票分組，並在每組內按日期排序
    work_df = work_df.sort_values(['stock_id', 'date'])
    
    # 使用 groupby 按股票分組，然後在每組內使用 rolling 按時間窗口計算相關性
    def calc_correlation(group):
        """計算單個股票組的相關性"""
        group_indexed = group.set_index('date').sort_index()
        return group_indexed[temp_col1].rolling(
            window=f'{window}D', min_periods=1
        ).corr(group_indexed[temp_col2])
    
    # 按股票分組計算，並合併結果
    result_list = []
    for stock_id, group in work_df.groupby('stock_id'):
        corr_values = calc_correlation(group)
        # 恢復原始索引
        corr_values.index = group.index
        result_list.append(corr_values)
    
    # 合併所有結果並按原始索引排序
    result = pd.concat(result_list).reindex(df.index)
    
    return result

def apply_covariance_by_stock(df, col1_or_series1, col2_or_series2, window):
    """
    按股票分組計算協方差（使用 date 索引優化版本）
    
    參數:
    df: DataFrame，包含 stock_id 和 date 欄位
    col1_or_series1, col2_or_series2: 要計算協方差的兩個欄位名稱或 Series 或 numpy array
    window: 窗口大小（以天為單位）
    
    返回:
    Series: 按股票分組並按日期索引計算的協方差
    """
    # 確保 DataFrame 有必要欄位
    if 'stock_id' not in df.columns:
        raise ValueError("DataFrame 必須包含 'stock_id' 欄位")
    if 'date' not in df.columns:
        raise ValueError("DataFrame 必須包含 'date' 欄位")
    
    # 創建 DataFrame 副本以避免修改原始數據
    work_df = df.copy()
    
    # 生成臨時欄位名稱
    temp_col1 = '_temp_cov_col1_'
    temp_col2 = '_temp_cov_col2_'
    
    # 準備兩個要計算協方差的 Series，並暫時加入 DataFrame
    if isinstance(col1_or_series1, str):
        work_df[temp_col1] = work_df[col1_or_series1]
    else:
        # 如果是 Series 或 numpy array，轉換並加入 DataFrame
        if hasattr(col1_or_series1, 'reindex'):
            # 如果是 pandas Series
            if work_df.index.is_unique:
                work_df[temp_col1] = col1_or_series1.reindex(work_df.index)
            else:
                if len(col1_or_series1) == len(work_df):
                    work_df[temp_col1] = pd.Series(col1_or_series1, index=work_df.index).values
                else:
                    work_df[temp_col1] = pd.Series(col1_or_series1[:len(work_df)], index=work_df.index).values
        else:
            # 如果是 numpy array
            if len(col1_or_series1) == len(work_df):
                work_df[temp_col1] = col1_or_series1
            else:
                work_df[temp_col1] = col1_or_series1[:len(work_df)]
    
    if isinstance(col2_or_series2, str):
        work_df[temp_col2] = work_df[col2_or_series2]
    else:
        # 如果是 Series 或 numpy array，轉換並加入 DataFrame
        if hasattr(col2_or_series2, 'reindex'):
            # 如果是 pandas Series
            if work_df.index.is_unique:
                work_df[temp_col2] = col2_or_series2.reindex(work_df.index)
            else:
                if len(col2_or_series2) == len(work_df):
                    work_df[temp_col2] = pd.Series(col2_or_series2, index=work_df.index).values
                else:
                    work_df[temp_col2] = pd.Series(col2_or_series2[:len(work_df)], index=work_df.index).values
        else:
            # 如果是 numpy array
            if len(col2_or_series2) == len(work_df):
                work_df[temp_col2] = col2_or_series2
            else:
                work_df[temp_col2] = col2_or_series2[:len(work_df)]
    
    # 確保 date 欄位為 datetime 格式並設置為索引，同時保持股票分組
    if not pd.api.types.is_datetime64_any_dtype(work_df['date']):
        work_df['date'] = pd.to_datetime(work_df['date'])
    
    # 按股票分組，並在每組內按日期排序
    work_df = work_df.sort_values(['stock_id', 'date'])
    
    # 使用 groupby 按股票分組，然後在每組內使用 rolling 按時間窗口計算協方差
    def calc_covariance(group):
        """計算單個股票組的協方差"""
        group_indexed = group.set_index('date').sort_index()
        return group_indexed[temp_col1].rolling(
            window=f'{window}D', min_periods=1
        ).cov(group_indexed[temp_col2])
    
    # 按股票分組計算，並合併結果
    result_list = []
    for stock_id, group in work_df.groupby('stock_id'):
        cov_values = calc_covariance(group)
        # 恢復原始索引
        cov_values.index = group.index
        result_list.append(cov_values)
    
    # 合併所有結果並按原始索引排序
    result = pd.concat(result_list).reindex(df.index)
    
    return result

# Alpha101 因子實現
def alpha_001(df):
    """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    # 按股票分組計算 stddev(returns, 20)
    stddev_returns = apply_ts_function_by_stock(df, 'returns', ts_stddev, 20)
    
    # 使用 numpy.where 進行向量化條件賦值
    condition = df['returns'] < 0
    signed_power_val = np.where(condition, 
                               signed_power(stddev_returns, 2), 
                               signed_power(df['close'], 2))
    
    # 按股票分組計算 ts_argmax
    ts_argmax_result = apply_ts_function_by_stock(df, signed_power_val, ts_argmax, 5)
    
    # 橫截面排名
    return cross_sectional_rank(df, ts_argmax_result) - 0.5

def alpha_002(df):
    """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    # 按股票分組計算 delta(log(volume), 2)
    delta_log_volume = apply_ts_function_by_stock(df, np.log(df['volume']), delta, 2, period=2)
    rank_delta_log_volume = cross_sectional_rank(df, delta_log_volume)
    rank_close_open_ratio = cross_sectional_rank(df, (df['close'] - df['open']) / df['open'])
    
    # 按股票分組計算相關性
    corr_result = apply_correlation_by_stock(df, rank_delta_log_volume, rank_close_open_ratio, 6)
    return -1 * corr_result

def alpha_003(df):
    """(-1 * correlation(rank(open), rank(volume), 10))"""
    rank_open = cross_sectional_rank(df, df['open'])
    rank_volume = cross_sectional_rank(df, df['volume'])
    
    # 按股票分組計算相關性
    corr_result = apply_correlation_by_stock(df, rank_open, rank_volume, 10)
    return -1 * corr_result

def alpha_004(df):
    """(-1 * Ts_Rank(rank(low), 9))"""
    # 先計算橫截面排名
    rank_low = cross_sectional_rank(df, df['low'])
    # 按股票分組計算 ts_rank
    ts_rank_result = apply_ts_function_by_stock(df, rank_low, ts_rank, 9)
    return -1 * ts_rank_result

def alpha_005(df):
    """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
    # 按股票分組計算 sum(vwap, 10)
    vwap_sum = apply_ts_function_by_stock(df, 'vwap', ts_sum, 10) / 10
    return cross_sectional_rank(df, df['open'] - vwap_sum) * (-1 * abs(cross_sectional_rank(df, df['close'] - df['vwap'])))

def alpha_006(df):
    """(-1 * correlation(open, volume, 10))"""
    # 按股票分組計算相關性
    corr_result = apply_correlation_by_stock(df, 'open', 'volume', 10)
    return -1 * corr_result

def alpha_007(df):
    """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
    # 按股票分組計算 adv20
    adv20 = apply_ts_function_by_stock(df, 'volume', ts_mean, 20)
    condition = adv20 < df['volume']
    
    # 按股票分組計算 delta(close, 7)
    delta_close_7 = apply_ts_function_by_stock(df, 'close', delta, 7, period=7)
    
    # 按股票分組計算 ts_rank(abs(delta(close, 7)), 60)
    ts_rank_result = apply_ts_function_by_stock(df, abs(delta_close_7), ts_rank, 60)
    
    # 使用 numpy.where 進行向量化條件賦值
    result = np.where(condition, 
                     (-1 * ts_rank_result) * np.sign(delta_close_7), 
                     -1)
    
    return result

def alpha_008(df):
    """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
    # 按股票分組計算 sum(open, 5) 和 sum(returns, 5)
    sum_open_5 = apply_ts_function_by_stock(df, 'open', ts_sum, 5)
    sum_returns_5 = apply_ts_function_by_stock(df, 'returns', ts_sum, 5)
    product = sum_open_5 * sum_returns_5
    
    # 按股票分組計算 delay(product, 10)
    delay_product = apply_ts_function_by_stock(df, product, delay, 10, d=10)
    
    return -1 * cross_sectional_rank(df, product - delay_product)

def alpha_009(df):
    """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
    # 按股票分組計算 delta(close, 1)
    delta_close_1 = apply_ts_function_by_stock(df, 'close', delta, 1, period=1)
    
    # 按股票分組計算 ts_min 和 ts_max
    ts_min_delta = apply_ts_function_by_stock(df, delta_close_1, ts_min, 5)
    ts_max_delta = apply_ts_function_by_stock(df, delta_close_1, ts_max, 5)
    
    condition1 = 0 < ts_min_delta
    condition2 = ts_max_delta < 0
    
    # 使用 numpy.where 進行向量化條件賦值
    result = np.where(condition1, delta_close_1,
                     np.where(condition2, delta_close_1, -1 * delta_close_1))
    
    return result

def alpha_010(df):
    """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
    # 按股票分組計算 delta(close, 1)
    delta_close_1 = apply_ts_function_by_stock(df, 'close', delta, 1, period=1)
    
    # 按股票分組計算 ts_min 和 ts_max
    ts_min_delta = apply_ts_function_by_stock(df, delta_close_1, ts_min, 4)
    ts_max_delta = apply_ts_function_by_stock(df, delta_close_1, ts_max, 4)
    
    condition1 = 0 < ts_min_delta
    condition2 = ts_max_delta < 0
    
    # 使用 numpy.where 進行向量化條件賦值
    result = np.where(condition1, delta_close_1,
                     np.where(condition2, delta_close_1, -1 * delta_close_1))
    
    return cross_sectional_rank(df, result)

def alpha_011(df):
    """((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
    vwap_close_diff = df['vwap'] - df['close']
    
    # 使用優化函數進行按股票分組計算
    ts_max_vwap_close = apply_ts_function_by_stock(df, vwap_close_diff, ts_max, 3)
    ts_min_vwap_close = apply_ts_function_by_stock(df, vwap_close_diff, ts_min, 3)
    delta_volume_3 = apply_ts_function_by_stock(df, 'volume', delta, 3, period=3)
    
    return (cross_sectional_rank(df, ts_max_vwap_close) + 
            cross_sectional_rank(df, ts_min_vwap_close)) * cross_sectional_rank(df, delta_volume_3)

def alpha_012(df):
    """(sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
    # 使用優化函數進行按股票分組計算
    delta_volume_1 = apply_ts_function_by_stock(df, 'volume', delta, 1, period=1)
    delta_close_1 = apply_ts_function_by_stock(df, 'close', delta, 1, period=1)
    
    return np.sign(delta_volume_1) * (-1 * delta_close_1)

def alpha_013(df):
    """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
    # 使用優化函數進行按股票分組計算
    rank_close = cross_sectional_rank(df, df['close'])
    rank_volume = cross_sectional_rank(df, df['volume'])
    
    cov_result = apply_covariance_by_stock(df, rank_close, rank_volume, 5)
    return -1 * cross_sectional_rank(df, cov_result)

def alpha_014(df):
    """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
    # 使用優化函數進行按股票分組計算
    delta_returns_3 = apply_ts_function_by_stock(df, 'returns', delta, 3, period=3)
    corr_open_volume = apply_correlation_by_stock(df, 'open', 'volume', 10)
    
    return (-1 * cross_sectional_rank(df, delta_returns_3)) * corr_open_volume

def alpha_015(df):
    """(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
    # 使用優化函數進行按股票分組計算
    rank_high = cross_sectional_rank(df, df['high'])
    rank_volume = cross_sectional_rank(df, df['volume'])
    
    corr_rank_high_volume = apply_correlation_by_stock(df, rank_high, rank_volume, 3)
    rank_corr = cross_sectional_rank(df, corr_rank_high_volume)
    
    return -1 * apply_ts_function_by_stock(df, rank_corr, ts_sum, 3)

def alpha_016(df):
    """(-1 * rank(covariance(rank(high), rank(volume), 5)))"""
    # 使用優化函數進行按股票分組計算
    rank_high = cross_sectional_rank(df, df['high'])
    rank_volume = cross_sectional_rank(df, df['volume'])
    
    cov_result = apply_covariance_by_stock(df, rank_high, rank_volume, 5)
    return -1 * cross_sectional_rank(df, cov_result)

def alpha_017(df):
    """(((rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
    adv20 = df['volume'].rolling(20).mean()
    # 按股票分組計算 ts_rank 和 delta
    ts_rank_close = apply_ts_function_by_stock(df, 'close', ts_rank, 10)
    delta_close_1 = apply_ts_function_by_stock(df, 'close', delta, 1, period=1)
    delta_delta_close = apply_ts_function_by_stock(df, delta_close_1, delta, 1, period=1)
    ts_rank_volume_adv20 = apply_ts_function_by_stock(df, df['volume'] / adv20, ts_rank, 5)
    
    return ((-1 * cross_sectional_rank(df, ts_rank_close)) * 
            cross_sectional_rank(df, delta_delta_close) * cross_sectional_rank(df, ts_rank_volume_adv20))

def alpha_018(df):
    """(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
    # 按股票分組計算 stddev 和 correlation
    stddev_abs_close_open = apply_ts_function_by_stock(df, abs(df['close'] - df['open']), ts_stddev, 5)
    close_open_diff = df['close'] - df['open']
    corr_close_open = apply_correlation_by_stock(df, 'close', 'open', 10)
    return -1 * cross_sectional_rank(df, stddev_abs_close_open + close_open_diff + corr_close_open)

def alpha_019(df):
    """((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    # 按股票分組計算 delay 和 delta
    close_delay_7 = apply_ts_function_by_stock(df, 'close', delay, 7, d=7)
    delta_close_7 = apply_ts_function_by_stock(df, 'close', delta, 7, period=7)
    sign_val = np.sign((df['close'] - close_delay_7) + delta_close_7)
    sum_returns_250 = apply_ts_function_by_stock(df, 'returns', ts_sum, 250)
    return (-1 * sign_val) * (1 + cross_sectional_rank(df, 1 + sum_returns_250))

def alpha_020(df):
    """(((rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
    # 按股票分組計算 delay
    high_delay_1 = apply_ts_function_by_stock(df, 'high', delay, 1, d=1)
    close_delay_1 = apply_ts_function_by_stock(df, 'close', delay, 1, d=1)
    low_delay_1 = apply_ts_function_by_stock(df, 'low', delay, 1, d=1)
    return (cross_sectional_rank(df, df['open'] - high_delay_1) * 
            cross_sectional_rank(df, df['open'] - close_delay_1) * 
            cross_sectional_rank(df, df['open'] - low_delay_1))

def alpha_021(df):
    """((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))"""
    adv20 = df['volume'].rolling(20).mean()
    # 按股票分組計算時間序列函數
    sum_close_8 = apply_ts_function_by_stock(df, 'close', ts_sum, 8) / 8
    sum_close_2 = apply_ts_function_by_stock(df, 'close', ts_sum, 2) / 2
    stddev_close_8 = apply_ts_function_by_stock(df, 'close', ts_stddev, 8)
    
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
    # 按股票分組計算相關性和時間序列函數
    corr_high_volume = apply_correlation_by_stock(df, 'high', 'volume', 5)
    delta_corr = apply_ts_function_by_stock(df, corr_high_volume, delta, 5, period=5)
    stddev_close_20 = apply_ts_function_by_stock(df, 'close', ts_stddev, 20)
    return -1 * delta_corr * cross_sectional_rank(df, stddev_close_20)

def alpha_023(df):
    """(((sum(high, 20) / 20) < high) ? (-1 * 1) : 1)"""
    # 按股票分組計算時間序列函數
    sum_high_20 = apply_ts_function_by_stock(df, 'high', ts_sum, 20) / 20
    condition = sum_high_20 < df['high']
    return np.where(condition, -1, 1)

def alpha_024(df):
    """(((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ? (-1 * (close - delay(close, 1))) : (-1 * delta(close, 3)))"""
    # 按股票分組計算時間序列函數
    sum_close_100 = apply_ts_function_by_stock(df, 'close', ts_sum, 100) / 100
    delta_sum_close = apply_ts_function_by_stock(df, sum_close_100, delta, 100, period=100)
    # 按股票分組計算 delay 和 delta
    close_delay_100 = apply_ts_function_by_stock(df, 'close', delay, 100, d=100)
    close_delay_1 = apply_ts_function_by_stock(df, 'close', delay, 1, d=1)
    delta_close_3 = apply_ts_function_by_stock(df, 'close', delta, 3, period=3)
    
    condition = (delta_sum_close / close_delay_100) < 0.05
    result = pd.Series(index=df.index)
    result[condition] = -1 * (df['close'] - close_delay_1)[condition]
    result[~condition] = -1 * delta_close_3[~condition]
    
    return result

def alpha_025(df):
    """rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
    adv20 = df['volume'].rolling(20).mean()
    return cross_sectional_rank(df, (-1 * df['returns']) * adv20 * df['vwap'] * (df['high'] - df['close']))

def alpha_026(df):
    """((((sum(close, 7) / 7) - close)) + (correlation(close, delay(close, 5), 230)))"""
    # 按股票分組計算時間序列函數和相關性
    sum_close_7 = apply_ts_function_by_stock(df, 'close', ts_sum, 7) / 7
    close_delay_5 = apply_ts_function_by_stock(df, 'close', delay, 5, d=5)
    corr_close_delay = apply_correlation_by_stock(df, 'close', close_delay_5, 230)
    return (sum_close_7 - df['close']) + corr_close_delay

def alpha_027(df):
    """WMA((close-delay(close,3))/delay(close,3)*100+(close-delay(close,6))/delay(close,6)*100,12)/100"""
    # 按股票分組計算 delay
    close_delay_3 = apply_ts_function_by_stock(df, 'close', delay, 3, d=3)
    close_delay_6 = apply_ts_function_by_stock(df, 'close', delay, 6, d=6)
    
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
    # 按股票分組計算 delay
    low_delay_1 = apply_ts_function_by_stock(df, 'low', delay, 1, d=1)
    high_delay_1 = apply_ts_function_by_stock(df, 'high', delay, 1, d=1)
    close_delay_1 = apply_ts_function_by_stock(df, 'close', delay, 1, d=1)
    
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
    # 按股票分組計算 delay
    close_delay_6 = apply_ts_function_by_stock(df, 'close', delay, 6, d=6)
    return (df['close'] - close_delay_6) / close_delay_6 * df['volume']

def alpha_030(df):
    """((rank(sign(delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
    # 按股票分組計算 delta
    delta_close_7 = apply_ts_function_by_stock(df, 'close', delta, 7, period=7)
    sign_delta = np.sign(delta_close_7)
    sum_returns_250 = apply_ts_function_by_stock(df, 'returns', ts_sum, 250)
    return cross_sectional_rank(df, sign_delta) * (1 + cross_sectional_rank(df, 1 + sum_returns_250))

def alpha_031(df):
    """((rank(rank(rank(decay_linear((-1 * rank(rank(rank(volume)))), 10)))) + rank((-1 * rank(returns, 10)))) + (rank(rank(abs(correlation(vwap, delay(close, 5), 10)))))"""
    # 分步驟計算，避免複雜的括號嵌套
    volume_rank1 = cross_sectional_rank(df, df['volume'])
    volume_rank2 = cross_sectional_rank(df, volume_rank1)
    volume_rank3 = cross_sectional_rank(df, volume_rank2)
    volume_ranked = -1 * volume_rank3
    
    # 按股票分組計算時間序列函數和相關性
    decay_vol = apply_ts_function_by_stock(df, volume_ranked, decay_linear, 10)
    rank_returns = apply_ts_function_by_stock(df, 'returns', ts_rank, 10)
    close_delay_5 = apply_ts_function_by_stock(df, 'close', delay, 5, d=5)
    vwap_delay_close = apply_correlation_by_stock(df, 'vwap', close_delay_5, 10)
    
    return cross_sectional_rank(df, cross_sectional_rank(df, cross_sectional_rank(df, decay_vol)) + cross_sectional_rank(df, -1 * rank_returns) + cross_sectional_rank(df, cross_sectional_rank(df, abs(vwap_delay_close))))

def alpha_032(df):
    """(scale(((sum(close, 7) / 7) - close)) + (scale(correlation(vwap, delay(close, 5), 230))))"""
    # 按股票分組計算時間序列函數和相關性
    sum_close_7 = apply_ts_function_by_stock(df, 'close', ts_sum, 7) / 7
    close_delay_5 = apply_ts_function_by_stock(df, 'close', delay, 5, d=5)
    vwap_delay_close = apply_correlation_by_stock(df, 'vwap', close_delay_5, 230)
    
    return scale(sum_close_7 - df['close']) + scale(vwap_delay_close)

def alpha_033(df):
    """rank((-1 * ((1 - (open / close)) * 1)))"""
    return cross_sectional_rank(df, -1 * (1 - (df['open'] / df['close'])) * 1)

def alpha_034(df):
    """rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    # 按股票分組計算時間序列函數
    stddev_returns_2 = apply_ts_function_by_stock(df, 'returns', ts_stddev, 2)
    stddev_returns_5 = apply_ts_function_by_stock(df, 'returns', ts_stddev, 5)
    delta_close_1 = apply_ts_function_by_stock(df, 'close', delta, 1, period=1)
    
    return cross_sectional_rank(df, (1 - cross_sectional_rank(df, stddev_returns_2 / stddev_returns_5)) + (1 - cross_sectional_rank(df, delta_close_1)))

def alpha_035(df):
    """((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    ts_rank_volume = apply_ts_function_by_stock(df, 'volume', ts_rank, 32)
    ts_rank_close_high_low = apply_ts_function_by_stock(df, df['close'] + df['high'] - df['low'], ts_rank, 16)
    ts_rank_returns = apply_ts_function_by_stock(df, 'returns', ts_rank, 32)
    
    return ts_rank_volume * (1 - ts_rank_close_high_low) * (1 - ts_rank_returns)

def alpha_036(df):
    """(((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5))) + rank(abs(correlation(vwap, adv20, 6))) + (0.6 * rank((((sum(close, 20) / 20) - open) * (close - open)))))"""
    adv20 = df['volume'].rolling(20).mean()
    close_open_diff = df['close'] - df['open']
    # 按股票分組計算 delay
    volume_delay_1 = apply_ts_function_by_stock(df, 'volume', delay, 1, d=1)
    returns_delay_6 = apply_ts_function_by_stock(df, -1 * df['returns'], delay, 6, d=6)
    sum_close_20 = apply_ts_function_by_stock(df, 'close', ts_sum, 20) / 20
    
    corr_close_open_volume = apply_correlation_by_stock(df, close_open_diff, volume_delay_1, 15)
    ts_rank_returns = apply_ts_function_by_stock(df, returns_delay_6, ts_rank, 5)
    corr_vwap_adv20 = apply_correlation_by_stock(df, 'vwap', adv20, 6)
    
    return (2.21 * cross_sectional_rank(df, corr_close_open_volume) + 
            0.7 * cross_sectional_rank(df, df['open'] - df['close']) + 
            0.73 * cross_sectional_rank(df, ts_rank_returns) + 
            cross_sectional_rank(df, abs(corr_vwap_adv20)) + 
            0.6 * cross_sectional_rank(df, (sum_close_20 - df['open']) * (df['close'] - df['open'])))

def alpha_037(df):
    """(rank(correlation(delta(close, 1), delay(delta(close, 1), 1), 3)) * rank((rank(correlation(rank(volume), rank(vwap), 6)))))"""
    # 按股票分組計算 delta 和 delay
    delta_close_1 = apply_ts_function_by_stock(df, 'close', delta, 1, period=1)
    delta_close_delay_1 = apply_ts_function_by_stock(df, delta_close_1, delay, 1, d=1)
    corr_delta = apply_correlation_by_stock(df, delta_close_1, delta_close_delay_1, 3)
    corr_volume_vwap = apply_correlation_by_stock(df, cross_sectional_rank(df, df['volume']), cross_sectional_rank(df, df['vwap']), 6)
    
    return cross_sectional_rank(df, corr_delta) * cross_sectional_rank(df, cross_sectional_rank(df, corr_volume_vwap))

def alpha_038(df):
    """((-1 * rank(Ts_Rank(returns, 10))) * rank((close / open)))"""
    ts_rank_returns = apply_ts_function_by_stock(df, 'returns', ts_rank, 10)
    close_open_ratio = df['close'] / df['open']
    
    return -1 * cross_sectional_rank(df, ts_rank_returns) * cross_sectional_rank(df, close_open_ratio)

def alpha_039(df):
    """((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    adv20 = df['volume'].rolling(20).mean()
    delta_close_7 = delta(df['close'], 7)
    decay_volume_adv20 = decay_linear(df['volume'] / adv20, 9)
    sum_returns_250 = apply_ts_function_by_stock(df, 'returns', ts_sum, 250)
    
    return -1 * cross_sectional_rank(df, delta_close_7 * (1 - cross_sectional_rank(df, decay_volume_adv20))) * (1 + cross_sectional_rank(df, sum_returns_250))

def alpha_040(df):
    """((-1 * rank(stddev(high, 10))) * correlation(close, volume, 10))"""
    # 按股票分組計算時間序列函數
    stddev_high_10 = apply_ts_function_by_stock(df, 'high', ts_stddev, 10)
    corr_close_volume = apply_correlation_by_stock(df, 'close', 'volume', 10)
    
    return -1 * cross_sectional_rank(df, stddev_high_10) * corr_close_volume

def alpha_041(df):
    """(((high * 0.6) + (vwap * 0.4)) - delay(((high * 0.6) + (vwap * 0.4)), 1))"""
    weighted_price = (df['high'] * 0.6) + (df['vwap'] * 0.4)
    # 按股票分組計算 delay
    weighted_price_delay_1 = apply_ts_function_by_stock(df, weighted_price, delay, 1, d=1)
    return weighted_price - weighted_price_delay_1

def alpha_042(df):
    """(rank(vwap - close) / rank(vwap + close))"""
    return cross_sectional_rank(df, df['vwap'] - df['close']) / cross_sectional_rank(df, df['vwap'] + df['close'])

def alpha_043(df):
    """(ts_rank((volume / adv20), 4) * ts_rank((-1 * delta(close, 7)), 4))"""
    adv20 = df['volume'].rolling(20).mean()
    volume_adv20_ratio = df['volume'] / adv20
    delta_close_7 = delta(df['close'], 7)
    
    return apply_ts_function_by_stock(df, volume_adv20_ratio, ts_rank, 4) * apply_ts_function_by_stock(df, -1 * delta_close_7, ts_rank, 4)

def alpha_044(df):
    """(-1 * correlation(high, rank(volume), 5))"""
    return -1 * apply_correlation_by_stock(df, 'high', cross_sectional_rank(df, df['volume']), 5)

def alpha_045(df):
    """(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    # 按股票分組計算 delay 和 ts_sum
    delay_close_5 = apply_ts_function_by_stock(df, 'close', delay, 5, d=5)
    sum_delay_close_20 = apply_ts_function_by_stock(df, delay_close_5, ts_sum, 20) / 20
    sum_close_5 = apply_ts_function_by_stock(df, 'close', ts_sum, 5)
    sum_close_20 = apply_ts_function_by_stock(df, 'close', ts_sum, 20)
    
    corr_close_volume = apply_correlation_by_stock(df, 'close', 'volume', 2)
    corr_sum_close = apply_correlation_by_stock(df, sum_close_5, sum_close_20, 2)
    
    return -1 * cross_sectional_rank(df, sum_delay_close_20) * corr_close_volume * cross_sectional_rank(df, corr_sum_close)

def alpha_046(df):
    """((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))"""
    # 按股票分組計算 delay
    delay_close_20 = apply_ts_function_by_stock(df, 'close', delay, 20, d=20)
    delay_close_10 = apply_ts_function_by_stock(df, 'close', delay, 10, d=10)
    delay_close_1 = apply_ts_function_by_stock(df, 'close', delay, 1, d=1)
    
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
    sum_high_5 = apply_ts_function_by_stock(df, 'high', ts_sum, 5) / 5
    
    term1 = (cross_sectional_rank(df, 1 / df['close']) * df['volume']) / adv20
    term2 = (df['high'] * cross_sectional_rank(df, df['high'] - df['close'])) / sum_high_5
    
    return term1 * term2

def alpha_048(df):
    """(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 3)) * rank(correlation(sum(volume, 5), sum(volume, 20), 2))), rank(sum(close, 5), 5)))"""
    # 簡化實現，移除 indneutralize 函數
    delta_close_1 = apply_ts_function_by_stock(df, 'close', delta, 1, period=1)
    close_delay_1 = apply_ts_function_by_stock(df, 'close', delay, 1, d=1)
    delta_delay_close_1 = apply_ts_function_by_stock(df, close_delay_1, delta, 1, period=1)
    sum_volume_5 = apply_ts_function_by_stock(df, 'volume', ts_sum, 5)
    sum_volume_20 = apply_ts_function_by_stock(df, 'volume', ts_sum, 20)
    sum_close_5 = apply_ts_function_by_stock(df, 'close', ts_sum, 5)
    
    corr_delta = apply_correlation_by_stock(df, delta_close_1, delta_delay_close_1, 3)
    corr_volume = apply_correlation_by_stock(df, sum_volume_5, sum_volume_20, 2)
    
    return corr_delta * cross_sectional_rank(df, corr_volume) * cross_sectional_rank(df, sum_close_5)

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
    corr_volume_vwap = correlation(cross_sectional_rank(df, df['volume']), cross_sectional_rank(df, df['vwap']), 5)
    return -1 * ts_max(cross_sectional_rank(df, corr_volume_vwap), 5)

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
    # 按股票分組計算所有ts函數
    ts_min_low_5 = apply_ts_function_by_stock(df, 'low', ts_min, 5)
    delay_ts_min_low_5 = apply_ts_function_by_stock(df, ts_min_low_5, delay, 5, d=5)
    sum_returns_240 = apply_ts_function_by_stock(df, 'returns', ts_sum, 240)
    sum_returns_20 = apply_ts_function_by_stock(df, 'returns', ts_sum, 20)
    
    term1 = (-1 * ts_min_low_5) + delay_ts_min_low_5
    term2 = cross_sectional_rank(df, (sum_returns_240 - sum_returns_20) / 220)
    term3 = apply_ts_function_by_stock(df, 'volume', ts_rank, 5)
    
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
    # 按股票分組計算 ts_min
    ts_min_low_12 = apply_ts_function_by_stock(df, 'low', ts_min, 12)
    term1 = (df['close'] - ts_min_low_12) ** 2
    corr_volume_vwap = apply_correlation_by_stock(df, 'volume', 'vwap', 6)
    
    return -1 * apply_correlation_by_stock(df, cross_sectional_rank(df, term1), cross_sectional_rank(df, corr_volume_vwap), 6)

def alpha_056(df):
    """(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
    # 按股票分組計算所有 ts_sum
    sum_returns_10 = apply_ts_function_by_stock(df, 'returns', ts_sum, 10)
    sum_returns_2 = apply_ts_function_by_stock(df, 'returns', ts_sum, 2)
    sum_sum_returns_2_3 = apply_ts_function_by_stock(df, sum_returns_2, ts_sum, 3)
    
    # 假設 cap 為 volume，因為原始公式中 cap 未定義
    cap = df['volume']
    
    return 0 - (1 * cross_sectional_rank(df, sum_returns_10 / sum_sum_returns_2_3) * cross_sectional_rank(df, df['returns'] * cap))

def alpha_057(df):
    """(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
    # 按股票分組計算 ts_argmax
    ts_argmax_close_30 = apply_ts_function_by_stock(df, 'close', ts_argmax, 30)
    decay_rank_ts_argmax = apply_ts_function_by_stock(df, cross_sectional_rank(df, ts_argmax_close_30), decay_linear, 2)
    
    return 0 - (1 * (df['close'] - df['vwap']) / decay_rank_ts_argmax)

def alpha_058(df):
    """(-1 * rank(((vwap * 0.728317) + (vwap * (1 - 0.728317)) - delay(vwap, 1))))"""
    weighted_vwap = (df['vwap'] * 0.728317) + (df['vwap'] * (1 - 0.728317))
    vwap_delay_1 = apply_ts_function_by_stock(df, 'vwap', delay, 1, d=1)
    return -1 * cross_sectional_rank(df, weighted_vwap - vwap_delay_1)

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
    # 按股票分組計算 ts_min
    ts_min_vwap = apply_ts_function_by_stock(df, 'vwap', ts_min, 16)
    adv20 = df['volume'].rolling(20).mean()
    corr_vwap_adv20 = apply_correlation_by_stock(df, 'vwap', adv20, 19)
    
    return rank(df['vwap'] - ts_min_vwap) < rank(corr_vwap_adv20)

def alpha_062(df):
    """((rank(correlation(vwap, sum(adv20, 14.4714), 6.4714)) + rank((open - delay(high, 1)))))"""
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = apply_ts_function_by_stock(df, adv20, ts_sum, 14)
    corr_vwap_sum_adv20 = apply_correlation_by_stock(df, 'vwap', sum_adv20, 6)
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
    sum_weighted_price = apply_ts_function_by_stock(df, weighted_price, ts_sum, 13)
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = apply_ts_function_by_stock(df, adv20, ts_sum, 13)
    corr_sum_price_adv20 = correlation(sum_weighted_price, sum_adv20, 17)
    
    close_open_weighted = (df['close'] * 0.955724) + (df['open'] * (1 - 0.955724))
    delta_close_open = delta(close_open_weighted, 3)
    
    return (rank(corr_sum_price_adv20) < rank(delta_close_open)) * -1

def alpha_065(df):
    """(rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv20, 6.6911), 6.6911)) < rank((open - delay(high, 1)))))"""
    weighted_open_vwap = (df['open'] * 0.00817205) + (df['vwap'] * (1 - 0.00817205))
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = apply_ts_function_by_stock(df, adv20, ts_sum, 7)
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
    ts_rank_decay = apply_ts_function_by_stock(df, decay_ratio, ts_rank, 7)
    
    return (rank(decay_delta_vwap) + ts_rank_decay) * -1

def alpha_067(df):
    """((rank((high - ts_min(high, 2.72412)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.sector), 6.4714)))"""
    # 簡化實現，移除 IndNeutralize 函數
    # 按股票分組計算 ts_min
    ts_min_high = apply_ts_function_by_stock(df, 'high', ts_min, 3)
    high_min_diff = df['high'] - ts_min_high
    adv20 = df['volume'].rolling(20).mean()
    corr_vwap_adv20 = apply_correlation_by_stock(df, 'vwap', adv20, 6)
    
    return rank(high_min_diff) ** rank(corr_vwap_adv20)

def alpha_068(df):
    """((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)"""
    adv15 = df['volume'].rolling(15).mean()
    corr_rank_high_adv15 = correlation(rank(df['high']), rank(adv15), 9)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_rank_high_adv15, ts_rank, 14)
    
    close_low_weighted = (df['close'] * 0.518371) + (df['low'] * (1 - 0.518371))
    delta_weighted = delta(close_low_weighted, 1)
    
    return (ts_rank_corr < rank(delta_weighted)) * -1

def alpha_069(df):
    """((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^rank(correlation(IndNeutralize(close, IndClass.industry), IndNeutralize(adv20, IndClass.industry), 4.79344)))"""
    # 簡化實現，移除 IndNeutralize 函數
    delta_vwap = delta(df['vwap'], 3)
    ts_max_delta = apply_ts_function_by_stock(df, delta_vwap, ts_max, 5)
    adv20 = df['volume'].rolling(20).mean()
    corr_close_adv20 = apply_correlation_by_stock(df, 'close', adv20, 5)
    
    return rank(ts_max_delta) ** rank(corr_close_adv20)

def alpha_070(df):
    """((rank(delta(vwap, 1.29456))^rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256)))"""
    # 簡化實現，移除 IndNeutralize 函數
    delta_vwap = delta(df['vwap'], 1)
    adv50 = df['volume'].rolling(50).mean()
    corr_close_adv50 = apply_correlation_by_stock(df, 'close', adv50, 18)
    
    return rank(delta_vwap) ** rank(corr_close_adv50)

def alpha_071(df):
    """max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.70376), Ts_Rank(adv15, 15.7152), 4.86682), 7.95524), 3.65999), Ts_Rank(decay_linear(Ts_Rank(correlation(rank(low), rank(adv15), 8.62541), 8.40009), 13.6813)))"""
    adv15 = df['volume'].rolling(15).mean()
    
    # 第一部分
    # 按股票分組計算 ts_rank
    ts_rank_close = apply_ts_function_by_stock(df, 'close', ts_rank, 4)
    ts_rank_adv15 = apply_ts_function_by_stock(df, adv15, ts_rank, 16)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_close, ts_rank_adv15, 5)
    decay_corr = apply_ts_function_by_stock(df, corr_ts_rank, decay_linear, 8)
    ts_rank_decay1 = apply_ts_function_by_stock(df, decay_corr, ts_rank, 4)
    
    # 第二部分
    corr_rank_low_adv15 = apply_correlation_by_stock(df, cross_sectional_rank(df, df['low']), cross_sectional_rank(df, adv15), 9)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_rank_low_adv15, ts_rank, 8)
    decay_ts_rank = apply_ts_function_by_stock(df, ts_rank_corr, decay_linear, 14)
    ts_rank_decay2 = apply_ts_function_by_stock(df, decay_ts_rank, ts_rank, 14)
    
    return np.maximum(ts_rank_decay1, ts_rank_decay2)

def alpha_072(df):
    """(rank(decay_linear(correlation(((high + low) / 2), adv20, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72465), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""
    high_low_avg = (df['high'] + df['low']) / 2
    adv20 = df['volume'].rolling(20).mean()
    # 按股票分組計算所有函數
    corr_hl_adv20 = apply_correlation_by_stock(df, high_low_avg, adv20, 9)
    decay_corr_hl = apply_ts_function_by_stock(df, corr_hl_adv20, decay_linear, 10)
    
    ts_rank_vwap = apply_ts_function_by_stock(df, 'vwap', ts_rank, 4)
    ts_rank_volume = apply_ts_function_by_stock(df, 'volume', ts_rank, 19)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_vwap, ts_rank_volume, 7)
    decay_corr_ts = apply_ts_function_by_stock(df, corr_ts_rank, decay_linear, 3)
    
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
    ts_rank_decay2 = apply_ts_function_by_stock(df, decay_delta_weighted, ts_rank, 9)
    
    return np.maximum(rank_decay1, ts_rank_decay2) * -1

def alpha_074(df):
    """((rank(correlation(close, sum(adv20, 37.0616), 15.1102)) < rank(correlation(rank(((high + low) / 2)), rank(volume), 11.2328))) * -1)"""
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = apply_ts_function_by_stock(df, adv20, ts_sum, 37)
    corr_close_sum_adv20 = apply_correlation_by_stock(df, 'close', sum_adv20, 15)
    
    high_low_avg = (df['high'] + df['low']) / 2
    corr_rank_hl_volume = apply_correlation_by_stock(df, cross_sectional_rank(df, high_low_avg), cross_sectional_rank(df, df['volume']), 11)
    
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
    ts_rank_corr = apply_ts_function_by_stock(df, corr_low_adv81, ts_rank, 7)
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
    sum_weighted = apply_ts_function_by_stock(df, low_vwap_weighted, ts_sum, 20)
    adv40 = df['volume'].rolling(40).mean()
    sum_adv40 = apply_ts_function_by_stock(df, adv40, ts_sum, 20)
    corr_sum_weighted_adv40 = correlation(sum_weighted, sum_adv40, 7)
    
    corr_rank_vwap_volume = correlation(rank(df['vwap']), rank(df['volume']), 6)
    
    return rank(corr_sum_weighted_adv40) ** rank(corr_rank_vwap_volume)

def alpha_079(df):
    """(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))"""
    # 簡化實現，移除 IndNeutralize 函數
    close_open_weighted = (df['close'] * 0.60733) + (df['open'] * (1 - 0.60733))
    # 按股票分組計算所有時間序列函數
    delta_weighted = apply_ts_function_by_stock(df, close_open_weighted, delta, 1, period=1)
    
    adv150 = df['volume'].rolling(150).mean()
    ts_rank_vwap = apply_ts_function_by_stock(df, 'vwap', ts_rank, 4)
    ts_rank_adv150 = apply_ts_function_by_stock(df, adv150, ts_rank, 9)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_vwap, ts_rank_adv150, 15)
    
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

def alpha_081(df):
    """((rank(Log(product(((close - open), (close - open)), 14.5864))) < rank(correlation(IndNeutralize(adv20, IndClass.industry), IndNeutralize(adv60, IndClass.industry), 8.4611)))"""
    # 簡化實現，移除 IndNeutralize 函數
    close_open_diff = df['close'] - df['open']
    product_diff = product(close_open_diff, 15)
    log_product = np.log(product_diff)
    rank_log = rank(log_product)
    
    adv20 = df['volume'].rolling(20).mean()
    adv60 = df['volume'].rolling(60).mean()
    corr_adv20_adv60 = correlation(adv20, adv60, 8)
    rank_corr = rank(corr_adv20_adv60)
    
    return rank_log < rank_corr

def alpha_082(df):
    """(min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283))"""
    # 簡化實現，移除 IndNeutralize 函數
    delta_open = delta(df['open'], 1)
    decay_delta = decay_linear(delta_open, 15)
    rank_decay1 = rank(decay_delta)
    
    open_weighted = (df['open'] * 0.634196) + (df['open'] * (1 - 0.634196))
    corr_volume_open = correlation(df['volume'], open_weighted, 17)
    decay_corr = decay_linear(corr_volume_open, 7)
    ts_rank_decay = apply_ts_function_by_stock(df, decay_corr, ts_rank, 13)
    
    return np.minimum(rank_decay1, ts_rank_decay)

def alpha_083(df):
    """((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))"""
    high_low_diff = df['high'] - df['low']
    # 按股票分組計算 ts_sum
    sum_close_5 = apply_ts_function_by_stock(df, 'close', ts_sum, 5) / 5
    ratio = high_low_diff / sum_close_5
    # 按股票分組計算 delay
    delay_ratio = apply_ts_function_by_stock(df, ratio, delay, 2, d=2)
    rank_delay = rank(delay_ratio)
    rank_volume = rank(rank(df['volume']))
    
    numerator = rank_delay * rank_volume
    denominator = ratio / (df['vwap'] - df['close'])
    
    return numerator / denominator

def alpha_084(df):
    """(SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796)))"""
    # 按股票分組計算 ts_max
    ts_max_vwap = apply_ts_function_by_stock(df, 'vwap', ts_max, 15)
    vwap_max_diff = df['vwap'] - ts_max_vwap
    # 按股票分組計算所有時間序列函數
    ts_rank_diff = apply_ts_function_by_stock(df, vwap_max_diff, ts_rank, 21)
    delta_close = apply_ts_function_by_stock(df, 'close', delta, 5, period=5)
    
    return signed_power(ts_rank_diff, delta_close)

def alpha_085(df):
    """(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))"""
    high_close_weighted = (df['high'] * 0.876703) + (df['close'] * (1 - 0.876703))
    adv30 = df['volume'].rolling(30).mean()
    # 按股票分組計算所有函數
    corr_weighted_adv30 = apply_correlation_by_stock(df, high_close_weighted, adv30, 10)
    
    high_low_avg = (df['high'] + df['low']) / 2
    ts_rank_hl = apply_ts_function_by_stock(df, high_low_avg, ts_rank, 4)
    ts_rank_volume = apply_ts_function_by_stock(df, 'volume', ts_rank, 10)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_hl, ts_rank_volume, 7)
    
    return rank(corr_weighted_adv30) ** rank(corr_ts_rank)

def alpha_086(df):
    """(Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open))))"""
    adv20 = df['volume'].rolling(20).mean()
    sum_adv20 = apply_ts_function_by_stock(df, adv20, ts_sum, 15)
    corr_close_sum = correlation(df['close'], sum_adv20, 6)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_close_sum, ts_rank, 20)
    
    open_close_sum = df['open'] + df['close']
    vwap_open_sum = df['vwap'] + df['open']
    diff = open_close_sum - vwap_open_sum
    rank_diff = rank(diff)
    
    return ts_rank_corr < rank_diff

def alpha_087(df):
    """(max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 2.72412), 2.72412)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4139)), 4.39567), 6.74588))"""
    # 簡化實現，移除 IndNeutralize 函數
    close_vwap_weighted = (df['close'] * 0.369701) + (df['vwap'] * (1 - 0.369701))
    delta_weighted = delta(close_vwap_weighted, 3)
    decay_delta = decay_linear(delta_weighted, 3)
    rank_decay1 = rank(decay_delta)
    
    adv81 = df['volume'].rolling(81).mean()
    corr_adv81_close = correlation(adv81, df['close'], 13)
    abs_corr = abs(corr_adv81_close)
    decay_abs = decay_linear(abs_corr, 4)
    ts_rank_decay = apply_ts_function_by_stock(df, decay_abs, ts_rank, 7)
    
    return np.maximum(rank_decay1, ts_rank_decay)

def alpha_088(df):
    """(rank(decay_linear(((rank(open, 6.47141) + rank(open, 14.4714)) - (rank(delay(close, 1), 6.47141) + rank(delay(close, 1), 14.4714))), 8.83606)))"""
    rank_open_6 = rank(df['open'])
    rank_open_14 = rank(df['open'])
    delay_close_1 = delay(df['close'], 1)
    rank_delay_close_6 = rank(delay_close_1)
    rank_delay_close_14 = rank(delay_close_1)
    
    term1 = rank_open_6 + rank_open_14
    term2 = rank_delay_close_6 + rank_delay_close_14
    diff = term1 - term2
    decay_diff = decay_linear(diff, 9)
    
    return rank(decay_diff)

def alpha_089(df):
    """(2 * (rank(decay_linear(correlation(((low + open) - (vwap + close)), delay(close, 1), 6.47141), 8.83606)) - rank(decay_linear(Ts_Rank(Ts_Rank(correlation(IndNeutralize(volume, IndClass.sector), ((open + close) / 2), 6.74425), 6.21089), 5.5375), 2.02764))))"""
    # 簡化實現，移除 IndNeutralize 函數
    low_open_sum = df['low'] + df['open']
    vwap_close_sum = df['vwap'] + df['close']
    diff1 = low_open_sum - vwap_close_sum
    delay_close_1 = delay(df['close'], 1)
    corr_diff_delay = correlation(diff1, delay_close_1, 6)
    decay_corr = decay_linear(corr_diff_delay, 9)
    rank_decay1 = rank(decay_corr)
    
    open_close_avg = (df['open'] + df['close']) / 2
    corr_volume_avg = correlation(df['volume'], open_close_avg, 7)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_volume_avg, ts_rank, 6)
    ts_rank_ts_rank = apply_ts_function_by_stock(df, ts_rank_corr, ts_rank, 6)
    decay_ts_rank = decay_linear(ts_rank_ts_rank, 5)
    rank_decay2 = rank(decay_ts_rank)
    
    return 2 * (rank_decay1 - rank_decay2)

def alpha_090(df):
    """((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856))"""
    # 簡化實現，移除 IndNeutralize 函數
    # 按股票分組計算 ts_max
    ts_max_close = apply_ts_function_by_stock(df, 'close', ts_max, 5)
    close_max_diff = df['close'] - ts_max_close
    rank_diff = rank(close_max_diff)
    
    adv40 = df['volume'].rolling(40).mean()
    # 按股票分組計算所有函數
    corr_adv40_low = apply_correlation_by_stock(df, adv40, 'low', 5)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_adv40_low, ts_rank, 3)
    
    return rank_diff ** ts_rank_corr

def alpha_091(df):
    """((Ts_Rank((close - ((high + low) / 2)), 9.06131) + Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), IndNeutralize(close, IndClass.subindustry), 9.74441), 17.7982)) < (rank(((max(open, 5))^2)) + Ts_Rank(decay_linear(((high + low) / 2), 19.0451), 7.06036)))"""
    # 簡化實現，移除 IndNeutralize 函數
    high_low_avg = (df['high'] + df['low']) / 2
    close_hl_diff = df['close'] - high_low_avg
    ts_rank_diff = apply_ts_function_by_stock(df, close_hl_diff, ts_rank, 9)
    
    adv40 = df['volume'].rolling(40).mean()
    corr_adv40_close = correlation(adv40, df['close'], 10)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_adv40_close, ts_rank, 18)
    
    term1 = ts_rank_diff + ts_rank_corr
    
    max_open_5 = np.maximum(df['open'], 5)
    max_open_squared = max_open_5 ** 2
    rank_max_open = rank(max_open_squared)
    
    decay_hl_avg = decay_linear(high_low_avg, 19)
    ts_rank_decay = apply_ts_function_by_stock(df, decay_hl_avg, ts_rank, 7)
    
    term2 = rank_max_open + ts_rank_decay
    
    return term1 < term2

def alpha_092(df):
    """min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (high + low)), 9.69216), 4.9759), Ts_Rank(decay_linear(correlation(Ts_Rank(low, 8.62591), Ts_Rank(adv30, 17.5206), 5.89503), 12.6052), 12.8281))"""
    high_low_avg = (df['high'] + df['low']) / 2
    hl_avg_close = high_low_avg + df['close']
    high_low_sum = df['high'] + df['low']
    condition = hl_avg_close < high_low_sum
    # 按股票分組計算所有時間序列函數
    decay_condition = apply_ts_function_by_stock(df, condition.astype(float), decay_linear, 10)
    ts_rank_decay1 = apply_ts_function_by_stock(df, decay_condition, ts_rank, 5)
    
    adv30 = df['volume'].rolling(30).mean()
    ts_rank_low = apply_ts_function_by_stock(df, 'low', ts_rank, 9)
    ts_rank_adv30 = apply_ts_function_by_stock(df, adv30, ts_rank, 18)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_low, ts_rank_adv30, 6)
    decay_corr = apply_ts_function_by_stock(df, corr_ts_rank, decay_linear, 13)
    ts_rank_decay2 = apply_ts_function_by_stock(df, decay_corr, ts_rank, 13)
    
    return np.minimum(ts_rank_decay1, ts_rank_decay2)

def alpha_093(df):
    """(Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.subindustry), ((low * 0.721544) + (low * (1 - 0.721544))), 6.00052), 5.2467), 2.88563) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(close, 8.62591), Ts_Rank(adv60, 17.5206), 8.43296), 8.17068), 6.02093), 4.88563))"""
    # 簡化實現，移除 IndNeutralize 函數
    low_weighted = (df['low'] * 0.721544) + (df['low'] * (1 - 0.721544))
    corr_volume_low = correlation(df['volume'], low_weighted, 6)
    decay_corr = decay_linear(corr_volume_low, 5)
    ts_rank_decay1 = apply_ts_function_by_stock(df, decay_corr, ts_rank, 3)
    
    adv60 = df['volume'].rolling(60).mean()
    # 按股票分組計算 ts_rank
    ts_rank_close = apply_ts_function_by_stock(df, 'close', ts_rank, 9)
    ts_rank_adv60 = apply_ts_function_by_stock(df, adv60, ts_rank, 18)
    corr_ts_rank = correlation(ts_rank_close, ts_rank_adv60, 8)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_ts_rank, ts_rank, 8)
    decay_ts_rank = decay_linear(ts_rank_corr, 6)
    ts_rank_decay2 = apply_ts_function_by_stock(df, decay_ts_rank, ts_rank, 5)
    
    return ts_rank_decay1 - ts_rank_decay2

def alpha_094(df):
    """((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 4.6566), 2.96556))"""
    # 按股票分組計算 ts_min
    ts_min_vwap = apply_ts_function_by_stock(df, 'vwap', ts_min, 12)
    vwap_min_diff = df['vwap'] - ts_min_vwap
    rank_diff = rank(vwap_min_diff)
    
    adv60 = df['volume'].rolling(60).mean()
    # 按股票分組計算 ts_rank
    ts_rank_vwap = apply_ts_function_by_stock(df, 'vwap', ts_rank, 20)
    # 按股票分組計算所有ts函數
    ts_rank_adv60 = apply_ts_function_by_stock(df, adv60, ts_rank, 4)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_vwap, ts_rank_adv60, 5)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_ts_rank, ts_rank, 3)
    
    return rank_diff ** ts_rank_corr

def alpha_095(df):
    """(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))"""
    # 按股票分組計算 ts_min
    ts_min_open = apply_ts_function_by_stock(df, 'open', ts_min, 12)
    open_min_diff = df['open'] - ts_min_open
    rank_diff = rank(open_min_diff)
    
    high_low_avg = (df['high'] + df['low']) / 2
    # 按股票分組計算 ts_sum
    sum_hl_avg = apply_ts_function_by_stock(df, high_low_avg, ts_sum, 19)
    adv40 = df['volume'].rolling(40).mean()
    sum_adv40 = apply_ts_function_by_stock(df, adv40, ts_sum, 19)
    # 按股票分組計算所有函數
    corr_sum = apply_correlation_by_stock(df, sum_hl_avg, sum_adv40, 13)
    rank_corr = rank(corr_sum)
    rank_corr_power = rank_corr ** 5
    ts_rank_power = apply_ts_function_by_stock(df, rank_corr_power, ts_rank, 12)
    
    return rank_diff < ts_rank_power

def alpha_096(df):
    """(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_Rank(((Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 2.72412)^5), 13.4663), 6.78068), 4.43831))"""
    # 簡化實現，移除 IndNeutralize 函數
    corr_rank_vwap_volume = correlation(rank(df['vwap']), rank(df['volume']), 4)
    decay_corr = decay_linear(corr_rank_vwap_volume, 4)
    ts_rank_decay1 = apply_ts_function_by_stock(df, decay_corr, ts_rank, 8)
    
    adv81 = df['volume'].rolling(81).mean()
    corr_low_adv81 = correlation(df['low'], adv81, 8)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_low_adv81, ts_rank, 3)
    ts_rank_power = ts_rank_corr ** 5
    ts_rank_power_rank = apply_ts_function_by_stock(df, ts_rank_power, ts_rank, 13)
    decay_ts_rank = decay_linear(ts_rank_power_rank, 7)
    ts_rank_decay2 = apply_ts_function_by_stock(df, decay_ts_rank, ts_rank, 4)
    
    return np.maximum(ts_rank_decay1, ts_rank_decay2)

def alpha_097(df):
    """(max(rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 6.7456)), Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 8.16123), Ts_Rank(adv60, 20.0711), 8.8926), 6.99510), 7.44019), 3.7154))"""
    # 簡化實現，移除 IndNeutralize 函數
    low_vwap_weighted = (df['low'] * 0.721001) + (df['vwap'] * (1 - 0.721001))
    # 按股票分組計算所有時間序列函數
    delta_weighted = apply_ts_function_by_stock(df, low_vwap_weighted, delta, 3, period=3)
    decay_delta = apply_ts_function_by_stock(df, delta_weighted, decay_linear, 7)
    rank_decay1 = rank(decay_delta)
    
    adv60 = df['volume'].rolling(60).mean()
    ts_rank_low = apply_ts_function_by_stock(df, 'low', ts_rank, 8)
    ts_rank_adv60 = apply_ts_function_by_stock(df, adv60, ts_rank, 20)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_low, ts_rank_adv60, 9)
    ts_rank_corr = apply_ts_function_by_stock(df, corr_ts_rank, ts_rank, 7)
    decay_ts_rank = apply_ts_function_by_stock(df, ts_rank_corr, decay_linear, 7)
    ts_rank_decay2 = apply_ts_function_by_stock(df, decay_ts_rank, ts_rank, 4)
    
    return np.maximum(rank_decay1, ts_rank_decay2)

def alpha_098(df):
    """(rank(decay_linear(correlation(vwap, sum(adv5, 26.4715), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_Rank(min(correlation(rank(open), rank(adv15), 20.0457), correlation(rank(high), rank(adv15), 20.0457)), 6.91477), 13.5723), 8.14799)))"""
    adv5 = df['volume'].rolling(5).mean()
    sum_adv5 = apply_ts_function_by_stock(df, adv5, ts_sum, 26)
    corr_vwap_sum = correlation(df['vwap'], sum_adv5, 5)
    decay_corr = decay_linear(corr_vwap_sum, 7)
    rank_decay1 = rank(decay_corr)
    
    adv15 = df['volume'].rolling(15).mean()
    corr_rank_open_adv15 = correlation(rank(df['open']), rank(adv15), 20)
    corr_rank_high_adv15 = correlation(rank(df['high']), rank(adv15), 20)
    min_corr = np.minimum(corr_rank_open_adv15, corr_rank_high_adv15)
    ts_rank_min = apply_ts_function_by_stock(df, min_corr, ts_rank, 7)
    ts_rank_ts_rank = apply_ts_function_by_stock(df, ts_rank_min, ts_rank, 14)
    decay_ts_rank = decay_linear(ts_rank_ts_rank, 8)
    rank_decay2 = rank(decay_ts_rank)
    
    return rank_decay1 - rank_decay2

def alpha_099(df):
    """((rank(correlation(sum(((close * 0.60833) + (open * (1 - 0.60833))), 9.06103), sum(adv60, 9.06103), 6.37221)) < rank(correlation(Ts_Rank(((high + low) / 2), 12.4667), Ts_Rank(volume, 11.1287), 6.45321)))"""
    close_open_weighted = (df['close'] * 0.60833) + (df['open'] * (1 - 0.60833))
    # 按股票分組計算 ts_sum
    sum_weighted = apply_ts_function_by_stock(df, close_open_weighted, ts_sum, 9)
    adv60 = df['volume'].rolling(60).mean()
    sum_adv60 = apply_ts_function_by_stock(df, adv60, ts_sum, 9)
    # 按股票分組計算所有函數
    corr_sum_weighted_adv60 = apply_correlation_by_stock(df, sum_weighted, sum_adv60, 6)
    rank_corr1 = rank(corr_sum_weighted_adv60)
    
    high_low_avg = (df['high'] + df['low']) / 2
    ts_rank_hl = apply_ts_function_by_stock(df, high_low_avg, ts_rank, 12)
    ts_rank_volume = apply_ts_function_by_stock(df, 'volume', ts_rank, 11)
    corr_ts_rank = apply_correlation_by_stock(df, ts_rank_hl, ts_rank_volume, 6)
    rank_corr2 = rank(corr_ts_rank)
    
    return rank_corr1 < rank_corr2

def alpha_100(df):
    """(0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - ((high + low) / 2)), IndClass.subindustry))) * (volume / adv20))))"""
    # 簡化實現，移除 IndNeutralize 函數
    numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
    denominator = df['high'] - df['low']
    ratio = numerator / denominator
    volume_ratio = ratio * df['volume']
    rank_volume_ratio = rank(volume_ratio)
    scale_rank = scale(rank_volume_ratio, 1.5)
    
    adv20 = df['volume'].rolling(20).mean()
    corr_close_rank_adv20 = correlation(df['close'], rank(adv20), 5)
    high_low_avg = (df['high'] + df['low']) / 2
    diff_corr_avg = corr_close_rank_adv20 - high_low_avg
    scale_diff = scale(diff_corr_avg)
    
    volume_adv20_ratio = df['volume'] / adv20
    
    return 0 - (1 * ((scale_rank - scale_diff) * volume_adv20_ratio))

def alpha_101(df):
    """((close - open) / delay(close, 1))"""
    # 按股票分組計算 delay
    delay_close_1 = apply_ts_function_by_stock(df, 'close', delay, 1, d=1)
    return (df['close'] - df['open']) / delay_close_1

# 完整的101個因子實現完成

# 因子字典（完整101個）
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
    'alpha_081': alpha_081,
    'alpha_082': alpha_082,
    'alpha_083': alpha_083,
    'alpha_084': alpha_084,
    'alpha_085': alpha_085,
    'alpha_086': alpha_086,
    'alpha_087': alpha_087,
    'alpha_088': alpha_088,
    'alpha_089': alpha_089,
    'alpha_090': alpha_090,
    'alpha_091': alpha_091,
    'alpha_092': alpha_092,
    'alpha_093': alpha_093,
    'alpha_094': alpha_094,
    'alpha_095': alpha_095,
    'alpha_096': alpha_096,
    'alpha_097': alpha_097,
    'alpha_098': alpha_098,
    'alpha_099': alpha_099,
    'alpha_100': alpha_100,
    'alpha_101': alpha_101,
}

def calc_alpha101_factors(df, vwap_window=10):
    """
    計算 Alpha101 因子（完整101個）
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
            # 直接計算因子，不使用分組
            df[name] = func(df)
        except Exception as e:
            print(f"計算 {name} 時發生錯誤: {e}")
            df[name] = np.nan
    
    return df


def calc_selected_alpha_factors(df, selected_indices, vwap_window=10):
    """
    根據輸入的 selected_indices（如 [1, 3, 5]），只計算指定的 Alpha 因子。
    傳回含所選 alpha_xxx 欄位的 DataFrame

    參數:
    df: DataFrame，需要包含必要的價格和成交量數據
    selected_indices: list，指定要計算的 alpha 編號（如 [1, 3, 5]）
    vwap_window: VWAP 計算窗口，預設為 10 天
    """
    df = df.copy()
    df.sort_values(['stock_id', 'date'], inplace=True)

    # 預先計算常用欄位
    if 'returns' not in df.columns:
        df['returns'] = df.groupby('stock_id')['close'].pct_change()

    # 如果沒有 vwap 欄位，自動計算
    if 'vwap' not in df.columns:
        df = add_vwap_to_df(df, window=vwap_window)

    # 構建所需的 alpha function 名稱
    selected_names = [f'alpha_{str(idx).zfill(3)}' for idx in selected_indices if f'alpha_{str(idx).zfill(3)}' in alpha_funcs]

    for name in tqdm(selected_names):
        func = alpha_funcs[name]
        try:
            # 直接計算因子
            df[name] = func(df)
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




