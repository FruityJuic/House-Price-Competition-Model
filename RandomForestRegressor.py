import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load Data
model_file_path = r"D:/file baru/mendata/machine learning competition/train.csv"
model_file_path_test = r"D:/file baru/mendata/machine learning competition/test.csv"

model_data = pd.read_csv(model_file_path)
model_data_test = pd.read_csv(model_file_path_test)

y = model_data.SalePrice
model_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
    '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 'FullBath', 
    'TotRmsAbvGrd', 'LotArea'
]

X = model_data[model_features]
test_X = model_data_test[model_features]

# 2. SPLIT DATA TERLEBIH DAHULU (Mencegah Data Leakage)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 3. IMPUTASI MEDIAN SECARA TERPISAH
# Gunakan median dari train_X saja untuk mengisi train_X, val_X, dan test_X
train_median = train_X.median()

train_X = train_X.fillna(train_median)
val_X = val_X.fillna(train_median)

# Imputasi data tes menggunakan median dari data train
test_X_clean = test_X.fillna(train_median)
X_clean = X.fillna(train_median)


total_kolom_Xclean = X_clean.isnull().sum()
total_seluruh_Xclean = np.prod(X_clean.shape)
persentase_kehilangan = (total_kolom_Xclean / total_seluruh_Xclean) * 100

print("===== PERSENTASE VALUE HILANG =====")
print(persentase_kehilangan)

# 4. Evaluasi Model Random Forest
# Catatan: n_estimators biasanya bagus di angka 100-300, tidak perlu dicari dari 2..100
model_rf = RandomForestRegressor(n_estimators=100, random_state=1)
model_rf.fit(train_X, train_y)
val_predicts = model_rf.predict(val_X)

mae = mean_absolute_error(val_y, val_predicts)
print(f"MAE Validasi Random Forest (Tanpa Leakage): {mae:.2f}")

# 5. Fit Model Akhir pada Seluruh Data (X_clean, y)
final_model = RandomForestRegressor(n_estimators=100, random_state=1)
final_model.fit(X_clean, y)
test_predict = final_model.predict(test_X_clean)

# Output Submit
output = pd.DataFrame({
    'Id': model_data_test['Id'],
    'SalePrice': test_predict
})
