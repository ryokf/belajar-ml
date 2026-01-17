from src.data_loader import load_and_clean_data, get_features_target
from src.model import train_linear_regression, evaluate_model, train_polynomial_regression, train_lasso_regression

def main():
    df = load_and_clean_data('data/car_data.csv')
    
    x, y = get_features_target(df)
    
    print("Hasil linier regression : ")
    model, X_test, y_test = train_linear_regression(x, y)
    evaluate_model(model, X_test, y_test)
    
    print("Hasil polynomial regression : ")
    model, X_test, y_test = train_polynomial_regression(x, y)
    evaluate_model(model, X_test, y_test)
    
    print("Hasil lasso regression : ")
    model, X_test, y_test = train_lasso_regression(x, y)
    evaluate_model(model, X_test, y_test)
    
    
if __name__ == "__main__":
    main()