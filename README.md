# DatathonVinUni-Hgiang
# 📊 Sales Forecasting with Time Series & Machine Learning

## 🚀 Overview

This project builds a robust time series forecasting model to predict **Revenue** and **COGS** using historical sales and web traffic data.
The solution leverages:

* 📅 Time-based feature engineering
* 🔁 Lag & rolling statistics
* 🌐 Web traffic signals (anti-leakage)
* ⚡ Gradient boosting via LightGBM
* 🔍 Model interpretability with SHAP

---

## 📂 Dataset

The model uses three main datasets:

* `sales.csv` → Historical revenue & COGS
* `web_traffic.csv` → Website behavior (sessions, bounce rate)
* `sample_submission.csv` → Submission format

---

## ⚙️ Feature Engineering

### 📅 Time Features

* Month, Day of Week, Day of Year, Week of Year
* Weekend indicator
* Cyclical encoding:

  * `sin/cos` transformations for seasonality

### 🎯 Tet Holiday Proxy

* A proxy feature to capture **Lunar New Year effects**
* Derived **only from date** (no external data → competition-safe)

### 🔁 Lag Features

* Revenue lags:

  * 7, 14, 30, 90, 365 days

### 📈 Rolling Statistics

* Moving averages and standard deviation:

  * Windows: 7, 30, 90 days

### 🌐 Web Traffic (Anti-Leakage)

* `sessions_lag_1`
* `bounce_rate_lag_1`

---

## 🧠 Model

We use **LightGBM Regressor** with:

* Objective: `regression_l1`
* Learning rate: `0.03`
* Leaves: `63`
* Early stopping for generalization

👉 Target transformation:

* Train on `log1p(Revenue)`
* Predict with `expm1()`

---

## 📊 Evaluation Metrics

The model is evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score
* MAPE (Mean Absolute Percentage Error)
* SMAPE (Symmetric MAPE)

---

## 🔍 Model Explainability

We use **SHAP** to:

* Understand feature importance
* Visualize model decisions

```python
shap.summary_plot(shap_values, X_val, plot_type="bar")
```

---

## 💰 COGS Estimation

Instead of predicting separately:

* Compute historical ratio:

```python
COGS / Revenue
```

* Use recent 30-day average to estimate future COGS

---

## 📤 Submission

Final output:

* `Revenue` → predicted via model
* `COGS` → derived from ratio

```bash
submission_final.csv
```

---

## 🏆 Key Highlights

✅ No external data used (competition-compliant)
✅ Anti-leakage feature design
✅ Strong seasonality modeling
✅ Interpretable ML pipeline
✅ Robust against overfitting

---

## 📌 Future Improvements

* TimeSeries Cross Validation (CV)
* Hyperparameter tuning (Optuna)
* Ensemble models (LightGBM + XGBoost)
* Advanced holiday modeling

---

## 👨‍💻 Author

**Nguyen (dakiohnguyen)**
📍 Data Science / Machine Learning

---

## ⭐ If you find this useful

Give this repo a ⭐ to support!
