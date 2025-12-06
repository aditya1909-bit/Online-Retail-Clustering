import pandas as pd
import os

INPUT_FILE = 'online_retail_II.xlsx'
OUTPUT_FILE = 'cleaned_online_retail.csv'

print(f"Loading data from {INPUT_FILE}...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"The input file {INPUT_FILE} does not exist.")

try:
    df = pd.read_excel(INPUT_FILE, sheet_name='Year 2009-2010', engine='openpyxl')
    print(f"Initial data shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()
    
print("Data Preprocessing...")

df.columns = df.columns.str.strip()

initial_rows = len(df)
df.dropna(subset=['Customer ID'], inplace=True)

print(f"Dropped {initial_rows - len(df)} rows with missing Customer ID.")

df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
print(f"Data shape after removing negative Quantity and Price: {df.shape}")

df['TotalAmount'] = df['Quantity'] * df['Price']

df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned data saved to {OUTPUT_FILE}")