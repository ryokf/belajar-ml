import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_csv("car_data.csv")

df_features = df[['Year', 'Kms_Driven', 'Fuel_Type']]
feature = pd.get_dummies(df_features, columns=['Fuel_Type'], drop_first=True)
target = df['Selling_Price']

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

print("Training dataset: ", x_train.shape)
print("Testing dataset: ", x_test.shape)

sns.pairplot(df, x_vars=['Year', 'Kms_Driven', 'Fuel_Type'], y_vars=['Selling_Price'], height=5, aspect=0.8, kind='scatter')
plt.show()

# linier regression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_test)

# for name, coef in zip(feature.columns, model.coef_):
#     print(f"Slope (Koefisien) untuk {name}: {coef:.5f}")
# print("intercept : ", model.intercept_)

# 1. Hitung Nilai Rapor (R2 Score)
# score = r2_score(y_test, y_pred)
# print(f"R2 Score (Akurasi): {score:.2f}")

# 2. Cek Contoh Asli vs Prediksi
# Kita sandingkan data asli dan tebakan komputer biar kelihatan bedanya
# hasil = pd.DataFrame({'Harga Asli': y_test, 'Prediksi AI': y_pred})
# print("\n--- 5 Contoh Tebakan Terakhir ---")
# print(hasil.head())