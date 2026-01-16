from src.data_loader import load_and_clean_data, get_features_target
from src.model import train_linear_regression, evaluate_model

def main():
    df = load_and_clean_data('data/car_data.csv')
    
    x, y = get_features_target(df)
    
    model, X_test, y_test = train_linear_regression(x, y)
    
    evaluate_model(model, X_test, y_test, x.columns)
    
if __name__ == "__main__":
    main()