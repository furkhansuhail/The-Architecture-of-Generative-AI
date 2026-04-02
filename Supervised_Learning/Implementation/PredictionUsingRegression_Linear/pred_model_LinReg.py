"""
=============================================================
  Stock Return Prediction using Linear Regression
  Domain: Finance / Economics
  Data: Yahoo Finance (via yfinance) / GBM Simulation
  Author: Advanced ML Project
=============================================================

Pipeline:
  1.  Data Acquisition
  2.  Feature Engineering (Technical Indicators)
  3.  Exploratory Data Analysis
  4.  Preprocessing (scaling, train/test split)
  5.  Linear Regression Model
  6.  Ridge & Lasso Comparison (Regularization)
  7.  Walk-Forward Validation (time-series aware)
  8.  Model Evaluation & Diagnostics
  9.  Future Price Forecasting (Next Week + Next Month)
  10. Visualization
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf  # used if network available; fallback to synthetic below

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import statsmodels.api as sm
from scipy import stats

# ── Cross-platform output directory (saves alongside this script) ──
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# ─────────────────────────────────────────────
# 1. DATA ACQUISITION
# ─────────────────────────────────────────────
print("=" * 60)
print("  STOCK RETURN PREDICTION — LINEAR REGRESSION PROJECT")
print("=" * 60)

TICKER = "AAPL"
START  = "2018-01-01"
END    = "2024-12-31"

print(f"\n[1] Generating realistic stock data for {TICKER} ({START} → {END})")
print(f"    Using Geometric Brownian Motion (GBM) — the Black-Scholes market model.")

np.random.seed(42)
trading_days = pd.bdate_range(START, END)
n            = len(trading_days)

S0    = 150.0
mu    = 0.12
sigma = 0.20
dt    = 1 / 252

Z       = np.random.standard_normal(n)
log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
close   = S0 * np.exp(np.cumsum(log_ret))

intraday_vol = sigma * np.sqrt(dt) * close
high   = close + np.abs(np.random.normal(0, intraday_vol * 0.5))
low    = close - np.abs(np.random.normal(0, intraday_vol * 0.5))
open_  = np.roll(close, 1); open_[0] = S0
vol_base = 5e7
volume = (vol_base + vol_base * 0.3 * np.random.randn(n)).clip(min=1e6).astype(int)

df = pd.DataFrame({
    "Open": open_, "High": high, "Low": low,
    "Close": close, "Volume": volume
}, index=trading_days)

print(f"    Generated {len(df):,} trading days. "
      f"Price range: ${df['Close'].min():.2f} – ${df['Close'].max():.2f}\n")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("[2] Engineering technical indicator features...")

def compute_features(frame):
    """Compute all technical indicator features on a DataFrame."""
    f = frame.copy()
    f["Return_1d"]  = f["Close"].pct_change(1)
    f["Return_5d"]  = f["Close"].pct_change(5)
    f["Return_21d"] = f["Close"].pct_change(21)
    f["SMA_10"]     = f["Close"].rolling(10).mean()
    f["SMA_50"]     = f["Close"].rolling(50).mean()
    f["EMA_12"]     = f["Close"].ewm(span=12).mean()
    f["EMA_26"]     = f["Close"].ewm(span=26).mean()
    f["Price_SMA10_ratio"] = f["Close"] / f["SMA_10"]
    f["Price_SMA50_ratio"] = f["Close"] / f["SMA_50"]
    f["MACD"]              = f["EMA_12"] - f["EMA_26"]
    f["Volatility_10d"]    = f["Return_1d"].rolling(10).std()
    f["Volatility_21d"]    = f["Return_1d"].rolling(21).std()
    delta = f["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["RSI_14"]      = 100 - (100 / (1 + rs))
    rm               = f["Close"].rolling(20).mean()
    rs2              = f["Close"].rolling(20).std()
    f["BB_Width"]    = (2 * rs2) / rm
    f["Volume_MA10"] = f["Volume"].rolling(10).mean()
    f["Volume_ratio"] = f["Volume"] / f["Volume_MA10"]
    f["HL_spread"]   = (f["High"] - f["Low"]) / f["Close"]
    return f

df = compute_features(df)
df["Target"] = df["Return_1d"].shift(-1)
df.dropna(inplace=True)
print(f"    {df.shape[1]-1} features engineered. {len(df):,} usable rows.\n")

# ─────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────
print("[3] Running Exploratory Data Analysis...")

FEATURES = [
    "Return_1d", "Return_5d", "Return_21d",
    "Price_SMA10_ratio", "Price_SMA50_ratio", "MACD",
    "Volatility_10d", "Volatility_21d",
    "RSI_14", "BB_Width", "Volume_ratio", "HL_spread"
]

X = df[FEATURES]
y = df["Target"]

print(f"\n    Target (Next-Day Return) Statistics:")
for label, val in [("Mean", y.mean()), ("Std", y.std()), ("Min", y.min()),
                   ("Max", y.max()), ("Skew", y.skew()), ("Kurtosis", y.kurtosis())]:
    print(f"    {label:>10}: {val:.6f}")

corr = X.corrwith(y).sort_values(key=abs, ascending=False)
print(f"\n    Feature Correlations with Target (top 5):")
for feat, val in corr.head(5).items():
    print(f"      {feat:<25}: {val:.4f}")

# ─────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────
print("\n[4] Preprocessing...")

SPLIT    = int(len(df) * 0.80)
X_train  = X.iloc[:SPLIT];  X_test  = X.iloc[SPLIT:]
y_train  = y.iloc[:SPLIT];  y_test  = y.iloc[SPLIT:]

scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

print(f"    Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# ─────────────────────────────────────────────
# 5. LINEAR REGRESSION
# ─────────────────────────────────────────────
print("\n[5] Training Linear Regression...")

lr = LinearRegression()
lr.fit(Xs_train, y_train)
y_pred_lr = lr.predict(Xs_test)

def evaluate(name, y_true, y_pred):
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    mae     = mean_absolute_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    print(f"\n    ── {name} ──")
    print(f"      RMSE          : {rmse:.6f}")
    print(f"      MAE           : {mae:.6f}")
    print(f"      R²            : {r2:.4f}")
    print(f"      Direction Acc : {dir_acc:.2%}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Dir_Acc": dir_acc}

results = {}
results["Linear Regression"] = evaluate("Linear Regression", y_test, y_pred_lr)

# ─────────────────────────────────────────────
# 6. RIDGE & LASSO
# ─────────────────────────────────────────────
print("\n[6] Training Ridge & Lasso (Regularization Comparison)...")

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.0001, max_iter=10000)
ridge.fit(Xs_train, y_train); lasso.fit(Xs_train, y_train)
y_pred_ridge = ridge.predict(Xs_test)
y_pred_lasso = lasso.predict(Xs_test)

results["Ridge (α=1.0)"]    = evaluate("Ridge (α=1.0)",    y_test, y_pred_ridge)
results["Lasso (α=0.0001)"] = evaluate("Lasso (α=0.0001)", y_test, y_pred_lasso)

# ─────────────────────────────────────────────
# 7. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────
print("\n[7] Walk-Forward Cross-Validation (5 folds)...")

tscv   = TimeSeriesSplit(n_splits=5)
Xs_all = scaler.fit_transform(X)  # refit for CV only
cv_scores = cross_val_score(LinearRegression(), Xs_all, y, cv=tscv, scoring="r2")
print(f"    R² per fold : {[f'{s:.4f}' for s in cv_scores]}")
print(f"    Mean R²     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Refit scaler on train only (restore for forecasting)
scaler.fit(X_train)

# ─────────────────────────────────────────────
# 8. STATSMODELS OLS
# ─────────────────────────────────────────────
print("\n[8] OLS Regression Summary (statsmodels)...")

X_sm = sm.add_constant(Xs_train)
ols  = sm.OLS(y_train, X_sm).fit()
print(f"\n    R-squared  : {ols.rsquared:.4f}")
print(f"    Adj R²     : {ols.rsquared_adj:.4f}")
print(f"    F-stat     : {ols.fvalue:.4f}  (p={ols.f_pvalue:.4e})")
print(f"\n    Significant features (p < 0.05):")
pvals = pd.Series(ols.pvalues[1:], index=FEATURES)
sig   = pvals[pvals < 0.05].sort_values()
for feat, pval in sig.items():
    coef = ols.params[FEATURES.index(feat) + 1]
    print(f"      {feat:<25}: coef={coef:+.6f}, p={pval:.4f}")
if sig.empty:
    print("      (none at 5% level — consistent with market efficiency)")

# ─────────────────────────────────────────────
# 9. FUTURE PRICE FORECASTING
# ─────────────────────────────────────────────
print("\n[9] Forecasting Future Prices (Next Week & Next Month)...")

FORECAST_WEEK  = 5   # trading days
FORECAST_MONTH = 21  # trading days
N_SIMULATIONS  = 1000  # Monte Carlo paths

# ── Historical residual std for noise injection ──
train_preds    = lr.predict(Xs_train)
residual_std   = np.std(y_train.values - train_preds)

def build_forecast_row(ohlcv_history):
    """
    Given a DataFrame of OHLCV history, compute the feature vector
    for the LAST row (most recent day) and return it as a 1-row array.
    """
    tmp = compute_features(ohlcv_history.copy())
    tmp.dropna(inplace=True)
    if tmp.empty:
        return None
    return tmp[FEATURES].iloc[[-1]]

def iterative_forecast(model, scaler, base_df, n_steps, noise_std=0.0, seed=None):
    """
    Roll the model forward n_steps days from the last row of base_df.

    Strategy:
      - Predict next-day return with the model
      - Apply it (plus optional noise) to get a new Close price
      - Reconstruct synthetic OHLCV for the new day
      - Recompute features on the extended history
      - Repeat

    Returns list of predicted prices (length n_steps).
    """
    if seed is not None:
        np.random.seed(seed)

    working = base_df.copy()
    prices  = []
    last_vol = working["Volume"].iloc[-10:].mean()

    for step in range(n_steps):
        row = build_forecast_row(working)
        if row is None:
            # Fall back to drift if we can't compute features
            prices.append(working["Close"].iloc[-1] * (1 + mu * dt))
            continue

        scaled_row     = scaler.transform(row)
        pred_return    = model.predict(scaled_row)[0]
        noise          = np.random.normal(0, noise_std) if noise_std > 0 else 0
        total_return   = pred_return + noise

        last_close  = working["Close"].iloc[-1]
        new_close   = last_close * (1 + total_return)
        daily_range = last_close * sigma * np.sqrt(dt)
        new_high    = new_close + abs(np.random.normal(0, daily_range * 0.5))
        new_low     = new_close - abs(np.random.normal(0, daily_range * 0.5))
        new_open    = last_close
        new_vol     = int(last_vol * np.random.uniform(0.85, 1.15))

        # Append new row
        next_date = working.index[-1] + pd.offsets.BDay(1)
        new_row   = pd.DataFrame({
            "Open":   [new_open],
            "High":   [new_high],
            "Low":    [new_low],
            "Close":  [new_close],
            "Volume": [new_vol]
        }, index=[next_date])
        working = pd.concat([working, new_row])
        prices.append(new_close)

    return prices

# ── Baseline (point forecast, no noise) ──
last_price       = df["Close"].iloc[-1]
last_date        = df.index[-1]
print(f"    Last known date  : {last_date.date()}")
print(f"    Last known price : ${last_price:.2f}")

week_prices_base  = iterative_forecast(lr, scaler, df, FORECAST_WEEK,  noise_std=0.0)
month_prices_base = iterative_forecast(lr, scaler, df, FORECAST_MONTH, noise_std=0.0)

week_price_forecast  = week_prices_base[-1]
month_price_forecast = month_prices_base[-1]

# ── Monte Carlo simulation for confidence intervals ──
print(f"    Running {N_SIMULATIONS:,} Monte Carlo simulations for confidence intervals...")

week_mc_final  = []
month_mc_final = []
week_mc_paths  = []
month_mc_paths = []

for sim in range(N_SIMULATIONS):
    wk = iterative_forecast(lr, scaler, df, FORECAST_WEEK,
                            noise_std=residual_std, seed=sim)
    mo = iterative_forecast(lr, scaler, df, FORECAST_MONTH,
                            noise_std=residual_std, seed=sim + N_SIMULATIONS)
    week_mc_final.append(wk[-1])
    month_mc_final.append(mo[-1])
    week_mc_paths.append(wk)
    month_mc_paths.append(mo)

week_mc_final  = np.array(week_mc_final)
month_mc_final = np.array(month_mc_final)
week_mc_paths  = np.array(week_mc_paths)
month_mc_paths = np.array(month_mc_paths)

# ── Confidence intervals (5th / 95th percentile = 90% CI) ──
def ci(arr, lo=5, hi=95):
    return np.percentile(arr, lo), np.percentile(arr, hi)

wk_lo,  wk_hi  = ci(week_mc_final)
mo_lo,  mo_hi  = ci(month_mc_final)

wk_path_lo  = np.percentile(week_mc_paths,  5,  axis=0)
wk_path_hi  = np.percentile(week_mc_paths,  95, axis=0)
mo_path_lo  = np.percentile(month_mc_paths, 5,  axis=0)
mo_path_hi  = np.percentile(month_mc_paths, 95, axis=0)

# ── Future date axes ──
week_dates  = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=FORECAST_WEEK)
month_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=FORECAST_MONTH)

# ── Pct change from last price ──
wk_chg  = (week_price_forecast  - last_price) / last_price * 100
mo_chg  = (month_price_forecast - last_price) / last_price * 100
wk_lo_c = (wk_lo  - last_price) / last_price * 100
wk_hi_c = (wk_hi  - last_price) / last_price * 100
mo_lo_c = (mo_lo  - last_price) / last_price * 100
mo_hi_c = (mo_hi  - last_price) / last_price * 100

print(f"\n    ┌─────────────────────────────────────────────────────┐")
print(f"    │              PRICE FORECAST SUMMARY                │")
print(f"    ├─────────────────────────────────────────────────────┤")
print(f"    │  Last Price   : ${last_price:>8.2f}                          │")
print(f"    ├─────────────────────────────────────────────────────┤")
print(f"    │  NEXT WEEK  ({FORECAST_WEEK} trading days → {week_dates[-1].date()})   │")
print(f"    │    Point Forecast : ${week_price_forecast:>8.2f}  ({wk_chg:+.2f}%)         │")
print(f"    │    90% CI         : ${wk_lo:>8.2f} – ${wk_hi:.2f}              │")
print(f"    │    CI Range       : {wk_lo_c:+.2f}% to {wk_hi_c:+.2f}%               │")
print(f"    ├─────────────────────────────────────────────────────┤")
print(f"    │  NEXT MONTH ({FORECAST_MONTH} trading days → {month_dates[-1].date()})  │")
print(f"    │    Point Forecast : ${month_price_forecast:>8.2f}  ({mo_chg:+.2f}%)         │")
print(f"    │    90% CI         : ${mo_lo:>8.2f} – ${mo_hi:.2f}              │")
print(f"    │    CI Range       : {mo_lo_c:+.2f}% to {mo_hi_c:+.2f}%               │")
print(f"    └─────────────────────────────────────────────────────┘")

# ─────────────────────────────────────────────
# 10. VISUALIZATION
# ─────────────────────────────────────────────
print("\n[10] Generating visualizations...")

fig = plt.figure(figsize=(22, 32))
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.48, wspace=0.35)

# ── Plot 1: Full price history ──
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df["Close"], color="#2196F3", linewidth=1.1, label="Close Price")
ax1.plot(df.index, df["SMA_10"], color="#FF9800", linewidth=0.8, linestyle="--", label="SMA-10", alpha=0.8)
ax1.plot(df.index, df["SMA_50"], color="#E91E63", linewidth=0.8, linestyle="--", label="SMA-50", alpha=0.8)
rm_p = df["Close"].rolling(20).mean()
rs_p = df["Close"].rolling(20).std()
ax1.fill_between(df.index, rm_p - 2*rs_p, rm_p + 2*rs_p,
                 alpha=0.12, color="#9C27B0", label="Bollinger Bands")
ax1.axvline(df.index[SPLIT], color="red", linestyle=":", linewidth=1.5, label="Train/Test Split")
ax1.set_title(f"{TICKER} — Closing Price with Technical Overlays", fontsize=14, fontweight="bold")
ax1.set_ylabel("Price (USD)"); ax1.legend(loc="upper left", fontsize=9)

# ── Plot 2: Return distribution ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(y, bins=80, color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.3)
ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
mu_y, sig_y = y.mean(), y.std()
xr = np.linspace(y.min(), y.max(), 200)
ax2.plot(xr, stats.norm.pdf(xr, mu_y, sig_y) * len(y) * (y.max()-y.min())/80,
         color="orange", linewidth=2, label="Normal fit")
ax2.set_title("Distribution of Next-Day Returns (Target)", fontweight="bold")
ax2.set_xlabel("Return"); ax2.set_ylabel("Frequency"); ax2.legend()

# ── Plot 3: Feature correlation heatmap ──
ax3 = fig.add_subplot(gs[1, 1])
corr_matrix = X[FEATURES].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, ax=ax3, cmap="RdBu_r", center=0,
            annot=False, linewidths=0.3, vmin=-1, vmax=1)
ax3.set_title("Feature Correlation Matrix", fontweight="bold")
ax3.tick_params(axis='x', rotation=45, labelsize=7)
ax3.tick_params(axis='y', labelsize=7)

# ── Plot 4: Actual vs Predicted returns ──
ax4 = fig.add_subplot(gs[2, 0])
s_n = min(200, len(y_test))
ax4.plot(range(s_n), y_test.values[:s_n], label="Actual",       color="#2196F3", alpha=0.8, linewidth=1)
ax4.plot(range(s_n), y_pred_lr[:s_n],    label="Predicted (LR)", color="#FF5722", alpha=0.8, linewidth=1, linestyle="--")
ax4.axhline(0, color="gray", linewidth=0.5)
ax4.set_title("Actual vs Predicted Returns (Test Set — first 200 days)", fontweight="bold")
ax4.set_xlabel("Trading Day"); ax4.set_ylabel("Return"); ax4.legend()

# ── Plot 5: Scatter actual vs predicted ──
ax5 = fig.add_subplot(gs[2, 1])
ax5.scatter(y_test, y_pred_lr, alpha=0.3, s=10, color="#673AB7")
lims = [min(y_test.min(), y_pred_lr.min()), max(y_test.max(), y_pred_lr.max())]
ax5.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax5.set_title("Actual vs Predicted — Scatter", fontweight="bold")
ax5.set_xlabel("Actual Return"); ax5.set_ylabel("Predicted Return"); ax5.legend()

# ── Plot 6: Residuals ──
residuals = y_test - y_pred_lr
ax6 = fig.add_subplot(gs[3, 0])
ax6.scatter(y_pred_lr, residuals, alpha=0.3, s=10, color="#009688")
ax6.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax6.set_title("Residuals vs Fitted Values", fontweight="bold")
ax6.set_xlabel("Fitted Values"); ax6.set_ylabel("Residuals")

# ── Plot 7: Model comparison ──
ax7 = fig.add_subplot(gs[3, 1])
model_names = list(results.keys())
rmse_vals   = [results[m]["RMSE"]    for m in model_names]
r2_vals     = [results[m]["R2"]      for m in model_names]
dir_vals    = [results[m]["Dir_Acc"] for m in model_names]
x_pos = np.arange(len(model_names))
w = 0.25
ax7.bar(x_pos - w, rmse_vals,            w, label="RMSE",          color="#2196F3", alpha=0.85)
ax7.bar(x_pos,     [abs(r) for r in r2_vals], w, label="|R²|",     color="#4CAF50", alpha=0.85)
ax7.bar(x_pos + w, dir_vals,             w, label="Direction Acc.", color="#FF9800", alpha=0.85)
ax7.set_title("Model Comparison: LR vs Ridge vs Lasso", fontweight="bold")
ax7.set_xticks(x_pos); ax7.set_xticklabels(model_names, fontsize=9)
ax7.legend(); ax7.set_ylabel("Metric Value")

# ── Plot 8: PRICE FORECAST — Next Week & Next Month ──
ax8 = fig.add_subplot(gs[4, :])

# Show last 40 trading days of actual price as context
context_days = 40
context_df   = df.iloc[-context_days:]

ax8.plot(context_df.index, context_df["Close"],
         color="#2196F3", linewidth=2.0, label="Actual Price (last 40 days)", zorder=5)

# Anchor point (connect last actual to forecast)
anchor_dates  = [last_date] + list(week_dates)
anchor_prices = [last_price] + week_prices_base
anchor_mo_dates  = [last_date] + list(month_dates)
anchor_mo_prices = [last_price] + month_prices_base

# ── Week forecast ──
ax8.plot(anchor_dates, anchor_prices,
         color="#FF5722", linewidth=2.2, linestyle="--",
         label=f"Week Forecast (Point)", zorder=6)

# Week CI band
ci_dates_wk  = [last_date] + list(week_dates)
ci_lo_wk     = [last_price] + list(wk_path_lo)
ci_hi_wk     = [last_price] + list(wk_path_hi)
ax8.fill_between(ci_dates_wk, ci_lo_wk, ci_hi_wk,
                 alpha=0.18, color="#FF5722", label="Week 90% CI")

# ── Month forecast ──
ax8.plot(anchor_mo_dates, anchor_mo_prices,
         color="#4CAF50", linewidth=2.2, linestyle="--",
         label=f"Month Forecast (Point)", zorder=6)

# Month CI band
ci_dates_mo  = [last_date] + list(month_dates)
ci_lo_mo     = [last_price] + list(mo_path_lo)
ci_hi_mo     = [last_price] + list(mo_path_hi)
ax8.fill_between(ci_dates_mo, ci_lo_mo, ci_hi_mo,
                 alpha=0.12, color="#4CAF50", label="Month 90% CI")

# Endpoint annotations — week
arrow_props = dict(arrowstyle="->", color="black", lw=1.2)
ax8.annotate(
    f"Week: ${week_price_forecast:.2f} ({wk_chg:+.1f}%)\n90% CI: ${wk_lo:.2f}–${wk_hi:.2f}",
    xy=(week_dates[-1], week_price_forecast),
    xytext=(week_dates[-1], week_price_forecast + (week_price_forecast * 0.025)),
    fontsize=9, color="#D84315", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCCBC", alpha=0.85),
    arrowprops=arrow_props
)

# Endpoint annotations — month
ax8.annotate(
    f"Month: ${month_price_forecast:.2f} ({mo_chg:+.1f}%)\n90% CI: ${mo_lo:.2f}–${mo_hi:.2f}",
    xy=(month_dates[-1], month_price_forecast),
    xytext=(month_dates[-1], month_price_forecast + (month_price_forecast * 0.025)),
    fontsize=9, color="#1B5E20", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#C8E6C9", alpha=0.85),
    arrowprops=arrow_props
)

# Vertical divider at last known date
ax8.axvline(last_date, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
ax8.text(last_date, ax8.get_ylim()[0] if ax8.get_ylim()[0] != 0 else last_price * 0.98,
         "  Forecast →", fontsize=8, color="gray", va="bottom")

ax8.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax8.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=30, ha="right")

ax8.set_title(
    f"{TICKER} — Price Forecast: Next Week ({week_dates[-1].date()}) "
    f"& Next Month ({month_dates[-1].date()})\n"
    f"Method: Iterative Linear Regression + {N_SIMULATIONS:,}-path Monte Carlo  |  "
    f"Shaded region = 90% Confidence Interval",
    fontsize=12, fontweight="bold"
)
ax8.set_ylabel("Price (USD)")
ax8.legend(loc="upper left", fontsize=9, ncol=2)
ax8.grid(True, alpha=0.4)

fig.suptitle(
    f"Stock Return Prediction — {TICKER} | Linear Regression Project",
    fontsize=16, fontweight="bold", y=0.995
)

out_chart = OUTPUT_DIR / "stock_return_prediction.png"
plt.savefig(out_chart, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Chart saved → {out_chart}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PROJECT SUMMARY")
print("=" * 60)
print(f"  Ticker        : {TICKER}")
print(f"  Date Range    : {START} → {END}")
print(f"  Features Used : {len(FEATURES)}")
print(f"  Train Size    : {len(X_train):,} days")
print(f"  Test Size     : {len(X_test):,} days")
print()
print(f"  {'Model':<25} {'RMSE':>10} {'R²':>8} {'Dir Acc':>10}")
print("  " + "-" * 55)
for m, r in results.items():
    print(f"  {m:<25} {r['RMSE']:>10.6f} {r['R2']:>8.4f} {r['Dir_Acc']:>10.2%}")
print()
print(f"  Walk-Forward CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()
print("  ─── PRICE FORECAST ─────────────────────────────────")
print(f"  Last Known Price   : ${last_price:.2f}  ({last_date.date()})")
print()
print(f"  Next Week  ({week_dates[-1].date()})")
print(f"    Predicted Price  : ${week_price_forecast:.2f}  ({wk_chg:+.2f}%)")
print(f"    90% Conf. Range  : ${wk_lo:.2f} – ${wk_hi:.2f}  "
      f"({wk_lo_c:+.1f}% to {wk_hi_c:+.1f}%)")
print()
print(f"  Next Month ({month_dates[-1].date()})")
print(f"    Predicted Price  : ${month_price_forecast:.2f}  ({mo_chg:+.2f}%)")
print(f"    90% Conf. Range  : ${mo_lo:.2f} – ${mo_hi:.2f}  "
      f"({mo_lo_c:+.1f}% to {mo_hi_c:+.1f}%)")
print()
print(f"  Monte Carlo Paths  : {N_SIMULATIONS:,}")
print(f"  Residual Std (noise): {residual_std:.6f}")
print("=" * 60)
print(f"\nDone! Outputs saved to: {OUTPUT_DIR}")