import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt 

# DATA PREPARATION
df = pd.read_csv("car_data2.csv")

df = df.rename(columns={"Prod. year" : "Year"})

df['Mileage'] = df["Mileage"].str.replace('km', '')
df['Mileage'] = pd.to_numeric(df['Mileage'])

df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
df['Levy'] = df['Levy'].fillna(0)
df['Levy'] = df['Levy'].astype('int64')

df['Doors'] = df['Doors'].str.replace(r'\D', '', regex=True)
df['Doors'] = df['Doors'].astype('int64')

df['Is Turbo'] = df['Engine volume'].str.contains('Turbo')
df['Engine volume'] = df['Engine volume'].str.replace('Turbo', '')
df['Engine volume'] = df['Engine volume'].astype('float')

df['Used'] = df['Mileage'] > 0

df['Cylinders'] = df['Cylinders'].astype('int64')

df = df[(df['Mileage'] > 0) & (df['Mileage'] < 500000 )]
df = df[(df['Price'] > 500) & (df['Price'] < 150000)]
df = df[(df['Engine volume'] < 10)]

df = df.drop(columns=['ID','Model', 'Wheel', 'Gear box type', 'Color', 'Doors', 'Drive wheels'])

df_feature = df[['Year', 'Engine volume', 'Mileage', 'Cylinders', 'Is Turbo', 'Manufacturer', 'Category', 'Fuel type', 'Used', 'Airbags', 'Leather interior']]
feature = pd.get_dummies(df_feature, columns=['Manufacturer', 'Category', 'Fuel type', 'Leather interior'], prefix=['Brand','Category','Fuel Type','Leather interior'])
target = df['Price']

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

# DATA VISUALIZATION
# sns.pairplot(df, x_vars=['Year', 'Engine volume', 'Mileage', 'Cylinders', 'Is Turbo', 'Manufacturer', 'Category', 'Fuel type'], y_vars=['Price'], height=5, aspect=0.8, kind='scatter')
sns.pairplot(df, x_vars=['Mileage'], y_vars=['Price'], height=5, aspect=0.8, kind='scatter')
# plt.show()

# DATA TRAINING
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

df_coef = pd.DataFrame({'Feature': feature.columns, 'Coefficient': model.coef_})
print(df_coef)

score = r2_score(y_test, y_pred)
print(f"R2 Score (Akurasi): {score:.2f}")
# print(df.info())