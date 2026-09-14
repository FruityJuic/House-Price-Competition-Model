import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# MACHINE LEARNING MODEL:DecisionTree. Model ini berfungsi untuk memprediksi harga rumah
# file dibagi dua, test dan train. train digunakan untuk melatih model menggunakan data train/latihan
# test digunakan sebagai data asli yang akan dijadikan sebagai data utama
# perbedaann, data train terdapat kolom 'harga' yang berguna sebagai acuan dalam memprediksi harga
# sedangkan data test tidak memiliki kolom 'harga'
# langkah-langkah terdapat di bawah

# 1. Ambil file path csv, baca file dan masukkan ke variabel 
model_file_path = r"D:\file baru\mendata\machine learning competition\train.csv"
model_file_path_test = r"D:\file baru\mendata\machine learning competition\test.csv"

model_data = pd.read_csv(model_file_path)
model_data_test = pd.read_csv(model_file_path_test)

y = model_data.SalePrice

model_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 
    '1stFlrSF', 'YearBuilt', 'YearRemodAdd', 'FullBath', 
    'TotRmsAbvGrd', 'LotArea'
]

X = model_data[model_features]
test_data_X = model_data_test[model_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#fungsi get_mae(menganalisis model dengan maksimal jumlah node)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X,train_y)
    predicst_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, predicst_val)
    return(mae)

candidate_max_leaf_nodes = []

for i in range(2, 200):
    candidate_max_leaf_nodes.append(i)
scores = {}

for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
  
    # simpan hasil leaf terbaik
    scores[max_leaf_nodes] = my_mae

best_size = min(scores, key=scores.get)

print(f"\n model dengan nodes terbaik adalah {best_size}")

# Fit ulang model akhir
final_model = DecisionTreeRegressor(max_leaf_nodes=best_size, random_state=1)
final_model.fit(train_X, train_y)
val_predict = final_model.predict(val_X)
galat = mean_absolute_error(val_y, val_predict)
print("===== MEAN ABSOLUTE ERROR MODEL ADALAH =====")
print(galat)

prediksi_akhir = final_model.predict(test_data_X)
data_awal = pd.DataFrame({
    'ID' : model_data_test['Id'],
    'SalePrice' : y
})

data_prediksi = pd.DataFrame({
    'ID': model_data_test['Id'],
    'SalePrice': prediksi_akhir
})

print("===== PERBANDINGAN DATA AWAL DENGAN DATA PREDIKSI =====")
print(data_awal.head())
print(data_prediksi.head())

data_prediksi.to_csv('Submission.csv', index=False)

