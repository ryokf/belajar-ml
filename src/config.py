CONFIG = {
    'data_path': 'data/car_data2.csv',
    'test_size': 0.2,
    'random_state': 42,
    'drop_cols': ['ID', 'Model', 'Wheel', 'Gear box type', 'Color', 'Doors', 'Drive wheels'],
    'numeric_cols': ['Levy', 'Doors', 'Cylinders'],
    'categorical_cols': ['Manufacturer', 'Category', 'Fuel type', 'Leather interior'],
    'feature_subset': [
        'Year', 'Engine volume', 'Mileage', 'Cylinders', 'Is Turbo', 
        'Manufacturer', 'Category', 'Fuel type', 'Used', 'Airbags', 'Leather interior'
    ]
}