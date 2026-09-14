# House-Price-Competition-Model
# 🏡 Kaggle House Price Prediction

A machine learning project predicting house prices using the Kaggle **House Prices: Advanced Regression Techniques** dataset. 

This project demonstrates clean data preprocessing, model tuning, and preventing data leakage to achieve accurate predictions.

---

## 💡 What's Inside

- **Data Cleaning:** Handled missing values using median imputation.
- **Data Leakage Prevention:** Split data into training and validation sets *before* cleaning to ensure honest evaluation.
- **Model Comparison:** Evaluated and tuned **Decision Tree** and **Random Forest** regressors.
- **Metric:** Evaluated performance using **Mean Absolute Error (MAE)**.

---

## 🛠️ Built With

- **Python**
- **Pandas** (Data manipulation)
- **Scikit-Learn** (Machine learning algorithms & metrics)

---

## 📊 Key Features Used

The models predict house prices using top features like:
- `OverallQual` (Overall material and finish quality)
- `GrLivArea` (Above grade living area in sq ft)
- `GarageCars` (Garage size in car capacity)
- `TotalBsmtSF` (Total basement area)
- `YearBuilt` & `YearRemodAdd` (Construction and remodel dates)
- `FullBath` (Full bathrooms above grade)
- `TotRmsAbvGrd` (Total rooms above grade (does not include bathrooms)

---
