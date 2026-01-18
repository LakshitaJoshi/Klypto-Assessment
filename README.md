# Klypto-Assessment

## **Regime-Aware Intraday Trading System for NIFTY Derivatives**

### 📌 Project Overview

This project implements a **regime-aware intraday trading framework** for NIFTY derivatives using **probabilistic market regime detection** and **machine learning–based trade filtering**.

Instead of attempting continuous price prediction, the system focuses on:

* identifying favorable market regimes,
* executing trades selectively based on regime alignment,
* enhancing trade quality using supervised ML models,
* and analyzing high-impact trades to understand performance drivers.

The end-to-end pipeline covers **data engineering, feature creation, regime modeling, backtesting, machine learning, and post-trade analysis**.

### 📊 Data Description

* **Frequency:** 5-minute intervals
* **Instruments:**

  * NIFTY Spot (OHLCV)
  * NIFTY Futures (OHLCV + Open Interest)
  * NIFTY Options (ATM ± strikes, CE & PE)

#### Data Source

Synthetic data is used to:

* maintain full reproducibility,
* control missing values and noise,
* ensure perfect timestamp alignment across instruments,
* allow realistic execution of data-cleaning tasks.

---

### 🧪 Feature Engineering

Key features engineered include:

#### Technical Indicators

* EMA (5), EMA (15)
* Spot returns
* EMA gap (signal strength)

#### Derivatives & Volatility

* Average implied volatility
* IV spread (Call – Put)
* Put–Call Ratio (OI-based and volume-based)
* Futures basis

#### Options Greeks (ATM)

* Delta, Gamma, Vega, Theta, Rho
  (calculated using the Black–Scholes model)

#### Risk & Exposure Metrics

* Average True Range (ATR)
* Gamma exposure
* Delta-neutral ratio
* Lagged returns (for ML models)

---

### 📈 Market Regime Detection

* A **3-state Gaussian Hidden Markov Model (HMM)** is used to classify the market into:

  * Uptrend
  * Sideways
  * Downtrend

* Regimes are interpreted post-training using return and volatility characteristics.

* The model is trained on the first 70% of the data to avoid look-ahead bias.

---

### 📉 Trading Strategy

#### Baseline Strategy

* EMA (5/15) crossover logic
* Long trades only in uptrend regimes
* Short trades only in downtrend regimes
* No trades during sideways regimes
* Trades executed at the next candle open

#### Backtesting

* Time-based split: 70% training / 30% testing
* Metrics evaluated:

  * Total return
  * Sharpe & Sortino ratio
  * Max drawdown
  * Calmar ratio
  * Win rate
  * Profit factor
  * Average trade duration

---

### 🤖 Machine Learning Enhancement

#### Problem Definition

Binary classification to predict whether a trade signal will be profitable:

* **Target:** 1 → profitable, 0 → not profitable

#### Models Used

* **XGBoost**

  * Time-series cross-validation
  * Tabular feature-based learning
* **LSTM**

  * Input: last 10 candles
  * Architecture: LSTM → Dropout → Dense → Output

#### ML-Enhanced Backtest

Trades are executed only when the ML model predicts profitability with confidence > 0.5.
Performance is compared across:

* Baseline strategy
* XGBoost-filtered strategy
* LSTM-filtered strategy

---

### 🔍 High-Performance Trade Analysis

* Profitable outlier trades identified using a **3-sigma (Z-score > 3)** rule

* Comparative analysis performed on:

  * regime
  * volatility (IV, ATR)
  * EMA gap
  * option Greeks
  * PCR
  * trade duration
  * time of day

* Statistical validation using **Mann–Whitney U tests**

---

### 🧠 Key Results Summary

* Regime-aware filtering reduces low-quality trades
* XGBoost outperforms LSTM on structured intraday data
* ML accuracy remains in the 50–60% range, which is realistic for financial time series
* A small fraction of trades contributes disproportionately to total PnL
* High-impact trades align with trending regimes, higher volatility, and stronger signals

---

### ⚙️ Installation Instructions

```bash
git clone <repository_url>
cd <repository_name>

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

### ▶️ How to Run

Run notebooks in the following order:

1. `01_data_acquisition.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_regime_detection.ipynb`
5. `05_baseline_strategy.ipynb`
6. `06_ml_models.ipynb`
7. `07_outlier_analysis.ipynb`

Each notebook saves outputs (CSV files, models, plots) to the appropriate folders.

---

### 📦 Requirements

All dependencies are listed in `requirements.txt`.
Key libraries include:

* pandas, numpy
* scikit-learn
* xgboost
* tensorflow / keras
* hmmlearn
* py_vollib
* matplotlib, seaborn

---

### 📌 Notes

* The project emphasizes **methodology and robustness**, not overfitting or inflated metrics.
* Results are intentionally realistic and reproducible.
* The modular design allows easy extension to live data or real-time deployment.

