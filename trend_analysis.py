import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

CSV_PATH = "trends.csv"
FORECAST_HORIZON = 10
OUTPUT_DIR = "trends_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_trends(path):
    df = pd.read_csv(path)
    df.columns = ['date', 'interest']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['interest'] = pd.to_numeric(
        df['interest'].astype(str).str.replace('<1', '0'), errors='coerce'
    )
    df = df.dropna(subset=['interest'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    freq = pd.infer_freq(df.index)
    if freq:
        df = df.asfreq(freq, method='ffill')
    else:
        df = df.asfreq('MS', method='ffill')
    print(f"Loaded: {len(df)} records, {df.index.min().date()} — {df.index.max().date()}, freq={df.index.freq}")
    return df


def plot_timeseries(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['interest'], color='#378ADD', linewidth=1.2)
    ax.fill_between(df.index, df['interest'], alpha=0.15, color='#378ADD')
    ax.set_title('Google Trends: Interest Over Time (2005-2025)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Search Interest')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot 1 saved")


def test_stationarity(df):
    ts = df['interest']
    w = 12
    rm = ts.rolling(window=w).mean()
    rs = ts.rolling(window=w).std()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts.index, ts, label='Original', color='#378ADD', alpha=0.7)
    ax.plot(rm.index, rm, label=f'Rolling Mean ({w})', color='#E24B4A', linewidth=2)
    ax.plot(rs.index, rs, label=f'Rolling Std ({w})', color='#1D9E75', linewidth=2)
    ax.set_title('Stationarity: Rolling Mean & Std', fontsize=15, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_stationarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    r = adfuller(ts.dropna(), autolag='AIC')
    print(f"\nADF Test: stat={r[0]:.4f}, p={r[1]:.6f}")
    for k, v in r[4].items():
        print(f"  {k}: {v:.4f}")
    print(f"  -> {'STATIONARY' if r[1] < 0.05 else 'NOT stationary'}")
    print("Plot 2 saved")


def plot_autocorrelation(df):
    ts = df['interest'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(ts, lags=40, ax=axes[0], title='ACF')
    plot_pacf(ts, lags=40, ax=axes[1], title='PACF')
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_acf_pacf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot 3 saved")


def forecast(df, horizon=FORECAST_HORIZON):
    ts = df['interest'].dropna().astype(float)
    train = ts[:-horizon]
    test = ts[-horizon:]

    best_aic, best_lag = np.inf, 1
    for lag in range(1, min(25, len(train) // 3)):
        try:
            r = AutoReg(train, lags=lag).fit()
            if r.aic < best_aic:
                best_aic, best_lag = r.aic, lag
        except:
            continue

    ar = AutoReg(train, lags=best_lag).fit()
    ar_pred = ar.predict(start=len(train), end=len(train) + horizon - 1)

    try:
        hw = ExponentialSmoothing(train, trend='add', damped_trend=True).fit(optimized=True)
        hw_pred = hw.forecast(horizon)
        hw_ok = True
    except:
        hw_ok = False

    print(f"\nAR({best_lag}): MAE={mean_absolute_error(test, ar_pred[:len(test)]):.2f}")
    if hw_ok:
        print(f"HW:      MAE={mean_absolute_error(test, hw_pred[:len(test)]):.2f}")

    ar_full = AutoReg(ts, lags=best_lag).fit()
    ar_f = ar_full.predict(start=len(ts), end=len(ts) + horizon - 1)
    hw_f = None
    if hw_ok:
        hw_full = ExponentialSmoothing(ts, trend='add', damped_trend=True).fit(optimized=True)
        hw_f = hw_full.forecast(horizon)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(len(ts)), ts.values, color='#378ADD', linewidth=1.2, label='Historical')
    ax.axvline(x=len(train), color='gray', linestyle='--', alpha=0.5)
    fx = range(len(ts), len(ts) + horizon)
    ax.plot(fx, ar_f.values, color='#E24B4A', linewidth=2, marker='o', markersize=4, label=f'AR({best_lag})')
    if hw_f is not None:
        ax.plot(fx, hw_f.values, color='#1D9E75', linewidth=2, marker='s', markersize=4, label='Holt-Winters')
    ax.set_title(f'Forecast: Next {horizon} Periods', fontsize=15, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot 4 saved")

    for i in range(horizon):
        line = f"  t+{i+1}: AR={ar_f.values[i]:.1f}"
        if hw_f is not None:
            line += f"  HW={hw_f.values[i]:.1f}"
        print(line)


if __name__ == "__main__":
    df = load_trends(CSV_PATH)
    plot_timeseries(df)
    test_stationarity(df)
    plot_autocorrelation(df)
    forecast(df, FORECAST_HORIZON)
    print(f"\nPlots- {OUTPUT_DIR}/")