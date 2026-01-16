from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

def train_linear_regression(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)

    pd.options.display.float_format = '{:,.3f}'.format
    df_coef = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_.flatten()})
    print(df_coef.sort_values(by='Coefficient', ascending=False).to_string())

    plt.scatter(X_test['Mileage'], y_test, color='blue', label='actual')
    plt.plot(X_test['Mileage'], y_pred, color='red', label='predicted')
    plt.show()

    score = r2_score(y_test, y_pred)
    print(f"R2 Score (Akurasi): {score:.2f}")