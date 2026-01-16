import pandas as pd

def load_and_clean_data(dataset):
    try:
        df = pd.read_csv(dataset)
    except FileNotFoundError:
        raise FileNotFoundError("file tidak ditemukan")
    
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
    
    return df

def get_features_target(df):
    df_feature = df[
        ['Year', 'Engine volume', 'Mileage', 'Cylinders', 'Is Turbo', 'Manufacturer', 'Category', 'Fuel type', 'Used', 'Airbags', 'Leather interior']
    ]
    
    x = pd.get_dummies(
        df_feature, 
        columns=['Manufacturer', 'Category', 'Fuel type', 'Leather interior'], 
        prefix=['Brand','Category','Fuel Type','Leather interior']
    )
    
    y = df['Price']
    
    return x, y