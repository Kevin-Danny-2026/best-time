import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# 1. 页面基本配置
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="多周期形态量化选股器",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 多周期形态与趋势联合选股器")
st.caption("自动匹配：标注1（三周期同向）、标注2（双底/背离）、标注3（通道回踩）、标注4（三角形/箱体突破）")

# -----------------------------------------------------------------------------
# 2. 技术指标与周期转换辅助函数
# -----------------------------------------------------------------------------
def calc_macd(df, fast=12, slow=26, signal=9):
    """计算 MACD 指标 (DIF, DEA, HIST)"""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    dif = exp1 - exp2
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist

def calc_bollinger(df, window=20, num_std=2):
    """计算布林线 (下轨, 中轨, 上轨)"""
    mid = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    return lower, mid, upper

def calc_rsi(df, window=14):
    """计算 RSI 指标"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def resample_kline(df_daily, rule='W'):
    """将日线重采样为周线(W)或月线(ME)"""
    df = df_daily.copy()
    df_res = df.resample(rule, on='date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    return df_res

# -----------------------------------------------------------------------------
# 3. 核心选股逻辑（4类形态识别）
# -----------------------------------------------------------------------------
def check_label_1(df_d, df_w, df_m):
    if len(df_m) < 22 or len(df_w) < 22 or len(df_d) < 35:
        return False, ""

    _, mid_m, _ = calc_bollinger(df_m)
    m_mid_up = mid_m.iloc[-1] > mid_m.iloc[-2]
    m_price_valid = df_m['close'].iloc[-1] >= mid_m.iloc[-1] or df_m['low'].iloc[-1] <= mid_m.iloc[-1] * 1.02
    if not (m_mid_up and m_price_valid):
        return False, ""

    low_w, mid_w, _ = calc_bollinger(df_w)
    w_touch = df_w['low'].iloc[-1] <= mid_w.iloc[-1] * 1.01 or df_w['low'].iloc[-1] <= low_w.iloc[-1] * 1.01
    body = abs(df_w['close'].iloc[-1] - df_w['open'].iloc[-1])
    lower_shadow = min(df_w['open'].iloc[-1], df_w['close'].iloc[-1]) - df_w['low'].iloc[-1]
    is_bullish_engulf = (df_w['close'].iloc[-1] > df_w['open'].iloc[-2]) and (df_w['close'].iloc[-2] < df_w['open'].iloc[-2])
    is_long_lower_shadow = lower_shadow > 1.5 * body if body > 0 else True
    if not (w_touch and (is_bullish_engulf or is_long_lower_shadow)):
        return False, ""

    ma20_d = df_d['close'].rolling(20).mean()
    ma30_d = df_d['close'].rolling(30).mean()
    vol_ma20_d = df_d['volume'].rolling(20).mean()
    vol_breakout = df_d['volume'].iloc[-1] > 1.5 * vol_ma20_d.iloc[-1]
    price_breakout = df_d['close'].iloc[-1] > ma20_d.iloc[-1] and df_d['close'].iloc[-1] > ma30_d.iloc[-1]
    dif_d, dea_d, _ = calc_macd(df_d)
    d_macd_gc = (dif_d.iloc[-1] > dea_d.iloc[-1]) and (dif_d.iloc[-2] <= dea_d.iloc[-2])
    _, _, hist_w = calc_macd(df_w)
    w_hist_shorten = (hist_w.iloc[-1] < 0) and (hist_w.iloc[-1] > hist_w.iloc[-2])

    if vol_breakout and price_breakout and d_macd_gc and w_hist_shorten:
        return True, "1 (三周期同向突破)"
    return False, ""

def check_label_2(df_d, df_w, df_m):
    if len(df_m) < 25 or len(df_w) < 25 or len(df_d) < 40:
        return False, ""

    m_rsi = calc_rsi(df_m)
    m_dif, _, _ = calc_macd(df_m)
    m_low_new = df_m['low'].iloc[-1] == df_m['low'].iloc[-12:].min()
    m_rsi_no_new_low = m_rsi.iloc[-1] > m_rsi.iloc[-12:].min()
    m_dif_no_new_low = m_dif.iloc[-1] > m_dif.iloc[-12:].min()
    m_body = abs(df_m['close'].iloc[-1] - df_m['open'].iloc[-1])
    m_lower_shadow = min(df_m['open'].iloc[-1], df_m['close'].iloc[-1]) - df_m['low'].iloc[-1]
    m_long_shadow = m_lower_shadow > (df_m['high'].iloc[-1] - df_m['low'].iloc[-1]) * 0.5 if (df_m['high'].iloc[-1] - df_m['low'].iloc[-1]) > 0 else False
    if not ((m_low_new and (m_rsi_no_new_low or m_dif_no_new_low)) or m_long_shadow):
        return False, ""

    w_lows = df_w['low'].iloc[-20:].values
    min_idx2 = -2
    min_idx1 = -12
    low1 = w_lows[min_idx1]
    low2 = w_lows[min_idx2]
    is_double_bottom = abs(low1 - low2) / low1 < 0.03
    vol_shrink = df_w['volume'].iloc[min_idx2] < df_w['volume'].iloc[min_idx1]
    neckline = df_w['high'].iloc[min_idx1:min_idx2].max() if min_idx1 < min_idx2 else df_w['high'].iloc[-5].max()
    if not (is_double_bottom and vol_shrink):
        return False, ""

    vol_ma20 = df_d['volume'].rolling(20).mean()
    vol_breakout = df_d['volume'].iloc[-2] > 1.5 * vol_ma20.iloc[-2] or df_d['volume'].iloc[-1] > 1.5 * vol_ma20.iloc[-1]
    retest_hold = df_d['low'].iloc[-1] >= neckline * 0.98 and df_d['close'].iloc[-1] > neckline

    if vol_breakout and retest_hold:
        return True, "2 (底背离+W底突破)"
    return False, ""

def check_label_3(df_d, df_w, df_m):
    if len(df_m) < 25 or len(df_w) < 25 or len(df_d) < 30:
        return False, ""

    _, mid_m, _ = calc_bollinger(df_m)
    m_slope_flat = abs(mid_m.iloc[-1] - mid_m.iloc[-3]) / mid_m.iloc[-3] < 0.015
    m_stand_above = df_m['close'].iloc[-1] > mid_m.iloc[-1]
    if not (m_slope_flat and m_stand_above):
        return False, ""

    low_w, mid_w, up_w = calc_bollinger(df_w)
    breakout_up_w = (df_w['high'].iloc[-8:-2] > up_w.iloc[-8:-2]).any()
    retest_hold_w = (df_w['low'].iloc[-2:] >= mid_w.iloc[-2:] * 0.99).all() and (df_w['close'].iloc[-1] > mid_w.iloc[-1])
    if not (breakout_up_w and retest_hold_w):
        return False, ""

    vol_ma20 = df_d['volume'].rolling(20).mean()
    ma20_d = df_d['close'].rolling(20).mean()
    d_start = (df_d['volume'].iloc[-1] > 1.5 * vol_ma20.iloc[-1]) and (df_d['close'].iloc[-1] > ma20_d.iloc[-1])

    if d_start:
        return True, "3 (中轨走平+二次启动)"
    return False, ""

def check_label_4(df_d, df_w, df_m):
    if len(df_m) < 20 or len(df_w) < 25 or len(df_d) < 30:
        return False, ""

    recent_highs = df_w['high'].iloc[-15:-1].values
    recent_lows = df_w['low'].iloc[-15:-1].values
    max_resistance = recent_highs.max()
    near_highs_count = np.sum(recent_highs >= max_resistance * 0.97)
    low1 = recent_lows[:5].min()
    low2 = recent_lows[5:10].min()
    low3 = recent_lows[10:].min()
    lows_raising = low1 <= low2 <= low3
    w_vol_ma = df_w['volume'].rolling(10).mean()
    w_breakout = (df_w['close'].iloc[-1] > max_resistance) and (df_w['volume'].iloc[-1] > 1.6 * w_vol_ma.iloc[-1])

    if not (near_highs_count >= 2 and lows_raising and w_breakout):
        return False, ""

    m_vol_ma = df_m['volume'].rolling(10).mean()
    m_vol_expand = df_m['volume'].iloc[-1] > 1.4 * m_vol_ma.iloc[-1]
    dif_m, dea_m, _ = calc_macd(df_m)
    m_macd_gc_zero = (dif_m.iloc[-1] > dea_m.iloc[-1]) and (dif_m.iloc[-2] <= dea_m.iloc[-2]) and (abs(dif_m.iloc[-1]) < 0.8)

    if m_vol_expand or m_macd_gc_zero:
        return True, "4 (三角形/箱体突破)"
    return False, ""

def analyze_stock(symbol, name):
    try:
        df_daily = None
        for _ in range(2):
            try:
                df_daily = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                if df_daily is not None and len(df_daily) >= 150:
                    break
            except Exception:
                time.sleep(0.3)

        if df_daily is None or len(df_daily) < 150:
            return None

        df_daily = df_daily.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        })
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily = df_daily.sort_values('date').reset_index(drop=True)

        df_weekly = resample_kline(df_daily, rule='W')
        df_monthly = resample_kline(df_daily, rule='ME')

        is_l1, desc1 = check_label_1(df_daily, df_weekly, df_monthly)
        if is_l1:
            return {"股票代码": symbol, "股票名称": name, "标注等级": "1 (最优)", "模式说明": desc1, "最新价": df_daily['close'].iloc[-1]}

        is_l2, desc2 = check_label_2(df_daily, df_weekly, df_monthly)
        if is_l2:
            return {"股票代码": symbol, "股票名称": name, "标注等级": "2 (次优)", "模式说明": desc2, "最新价": df_daily['close'].iloc[-1]}

        is_l3, desc3 = check_label_3(df_daily, df_weekly, df_monthly)
        if is_l3:
            return {"股票代码": symbol, "股票名称": name, "标注等级": "3", "模式说明": desc3, "最新价": df_daily['close'].iloc[-1]}

        is_l4, desc4 = check_label_4(df_daily, df_weekly, df_monthly)
        if is_l4:
            return {"股票代码": symbol, "股票名称": name, "标注等级": "4", "模式说明": desc4, "最新价": df_daily['close'].iloc[-1]}

        return None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 4. 兼容海外 IP 的股票列表获取函数 (带多重备用方案)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)  # 缓存 24 小时
def get_stock_list_safe():
    """多通道获取股票列表，防止海外 IP 被封"""
    # 方案 1：最轻量的全 A 股代码与名称接口
    try:
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty:
            return list(zip(df['code'], df['name']))
    except Exception:
        pass

    # 方案 2：尝试常规全量接口
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            return list(zip(df['代码'], df['名称']))
    except Exception:
        pass

    # 方案 3：降级方案 - 获取中证 800 成分股
    try:
        st.warning("⚠️ 全量股票接口受到防刷限制，已自动降级为【中证800成分股】进行筛选。")
        df_csi = ak.index_stock_cons(symbol="000906")
        if df_csi is not None and not df_csi.empty:
            return list(zip(df_csi['品种代码'], df_csi['品种名称']))
    except Exception:
        pass

    return []

# -----------------------------------------------------------------------------
# 5. UI 界面与多线程处理
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 运行与筛选设置")
max_threads = st.sidebar.slider("并行计算线程数", min_value=2, max_value=20, value=5)
stock_limit = st.sidebar.number_input("扫描股票数量 limit (0为全量检测)", min_value=0, max_value=5000, value=200, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 标注说明
* **标注 1**：月线向上 + 周线回踩止跌 + 日线放量突破 (三周期共振)
* **标注 2**：月线底背离 + 周线W底/圆弧底 + 日线突破颈线
* **标注 3**：月线走平站上 + 周线突破回踩 + 日线二次启动
* **标注 4**：周线上升三角形/箱体突破 + 月线量能配合
""")

if st.button("🚀 开始形态检测筛选", type="primary"):
    with st.spinner("正在安全获取股票清单..."):
        stock_list = get_stock_list_safe()
        if stock_limit > 0 and len(stock_list) > stock_limit:
            stock_list = stock_list[:stock_limit]

    if not stock_list:
        st.error("无法连接行情数据源获取股票列表，请稍后再试。")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(stock_list)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_map = {
                executor.submit(analyze_stock, code, name): (code, name)
                for code, name in stock_list
            }
            
            for future in as_completed(future_map):
                res = future.result()
                if res:
                    results.append(res)
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"已扫描 {completed}/{total} 只股票...")

        progress_bar.empty()
        status_text.empty()

        if results:
            df_res = pd.DataFrame(results)
            st.success(f"筛选完成！共匹配出 {len(df_res)} 只满足形态标准的股票。")
            df_res = df_res.sort_values(by="标注等级")
            st.dataframe(df_res, use_container_width=True)

            csv = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下载结果 CSV",
                data=csv,
                file_name=f"technical_pattern_stocks_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.warning("当前选定的股票列表中未找到完全匹配上述四类复杂形态的股票。")
