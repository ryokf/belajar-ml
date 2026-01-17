from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_lasso_regression(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = Lasso(alpha=1, max_iter=10000)
    
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def train_polynomial_regression(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(), memory=None)
    
    model.fit(X_train,y_train)
    
    return model, X_test, y_test
    

def train_linear_regression(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('harga asli')
    plt.ylabel('harga prediksi')

    # plt.show()

    r2 = r2_score(y_test, y_pred)
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"R2 Score (Akurasi): {r2:.2f}")
    print(f"RMSE: {rmse_score:.2f}")