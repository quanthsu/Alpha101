# Alpha101 因子實現

## 概述

本專案實現了基於世坤（WorldQuant）標準定義的 Alpha101 因子。Alpha101 是一套經典的量化投資因子，包含 101 個不同的技術指標。

## 文件說明

- `alpha101.py`: 主要實現文件，包含前 20 個因子的完整實現
- `alpha101_complete.py`: 完整版本，包含更多因子的實現示例
- `README_Alpha101.md`: 本說明文件

## 基礎函數

### 時間序列函數
- `ts_rank(series, window)`: 時間序列排名
- `ts_argmax(series, window)`: 時間序列最大值位置
- `ts_argmin(series, window)`: 時間序列最小值位置
- `ts_sum(series, window)`: 時間序列求和
- `ts_mean(series, window)`: 時間序列平均值
- `ts_stddev(series, window)`: 時間序列標準差
- `ts_min(series, window)`: 時間序列最小值
- `ts_max(series, window)`: 時間序列最大值

### 統計函數
- `correlation(x, y, window)`: 相關性
- `covariance(x, y, window)`: 協方差
- `rank(series)`: 橫截面排名
- `delta(series, period)`: 差分

### 其他函數
- `delay(series, d)`: 延遲 d 天
- `decay_linear(series, window)`: 線性衰減加權平均
- `scale(series, k)`: 縮放
- `signed_power(x, a)`: 符號冪
- `calculate_vwap(df)`: 計算成交量加權平均價
- `calculate_adv(df, window)`: 計算平均日成交量
- `add_vwap_to_df(df)`: 為 DataFrame 添加 VWAP 欄位

## 已實現的因子

### Alpha #1
```python
(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
```

### Alpha #2
```python
(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
```

### Alpha #3
```python
(-1 * correlation(rank(open), rank(volume), 10))
```

### Alpha #4
```python
(-1 * Ts_Rank(rank(low), 9))
```

### Alpha #5
```python
(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
```

...（更多因子請參考代碼）

## 使用方式

### 1. 準備數據

您的 DataFrame 需要包含以下欄位：
- `stock_id`: 股票代碼
- `date`: 日期
- `open`: 開盤價
- `close`: 收盤價
- `high`: 最高價
- `low`: 最低價
- `volume`: 成交量
- `vwap`: 成交量加權平均價（可選，會自動計算）

### 2. 計算因子

```python
import pandas as pd
from alpha101 import calc_alpha101_factors, add_vwap_to_df, calculate_vwap

# 準備數據
df = your_dataframe  # 包含上述欄位的 DataFrame

# 方法1：自動計算 VWAP（推薦）
result = calc_alpha101_factors(df)

# 方法2：手動計算 VWAP
df_with_vwap = add_vwap_to_df(df)
result = calc_alpha101_factors(df_with_vwap)

# 方法3：單獨計算 VWAP
vwap_values = calculate_vwap(df)

# 查看結果
print(result.columns)
print(result.head())
```

### 3. 範例代碼

```python
# 範例數據（不需要預先提供 vwap）
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

# 計算因子（會自動計算 VWAP）
result = calc_alpha101_factors(df)

# 或者手動計算 VWAP
df_with_vwap = add_vwap_to_df(df)
print("VWAP 值:", df_with_vwap['vwap'].tolist())
```

## VWAP 計算功能

### 什麼是 VWAP？
VWAP（Volume Weighted Average Price）是成交量加權平均價，是一個重要的技術指標，用於衡量股票的平均交易價格。

### 計算方法
```python
# 典型價格 = (最高價 + 最低價 + 收盤價) / 3
typical_price = (high + low + close) / 3

# 價格成交量乘積 = 典型價格 × 成交量
price_volume = typical_price × volume

# VWAP = 累積價格成交量乘積 / 累積成交量
vwap = cumulative_price_volume / cumulative_volume
```

### 使用方式
```python
from alpha101 import calculate_vwap, add_vwap_to_df

# 單獨計算 VWAP
vwap_values = calculate_vwap(df)

# 為 DataFrame 添加 VWAP 欄位
df_with_vwap = add_vwap_to_df(df)

# 在因子計算中自動使用
result = calc_alpha101_factors(df)  # 會自動計算 VWAP
```

## 注意事項

1. **數據要求**: 確保數據按股票代碼和日期排序
2. **缺失值處理**: 函數會自動處理缺失值，但建議預先清理數據
3. **計算效率**: 對於大量數據，建議分批處理
4. **記憶體使用**: 計算所有因子會增加記憶體使用量
5. **VWAP 計算**: 如果數據中沒有 VWAP 欄位，會自動計算

## 完整 101 個因子

由於篇幅限制，當前實現只包含前 20 個因子。完整的 101 個因子可以：

1. 參考世坤官方文檔
2. 使用專業的量化庫如 `alphalens`、`jaqs`、`rqalpha` 等
3. 根據需要逐步添加更多因子

## 參考資料

- [世坤 Alpha101 論文](https://www.worldquant.com/research/101-formulaic-alphas)
- [Alpha101 因子定義](https://www.worldquant.com/research/101-formulaic-alphas)

## 授權

本代碼僅供學習和研究使用，請遵守相關授權條款。 