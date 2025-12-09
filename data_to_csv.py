#Aditya Dutta & Ege Gürsel

# Load the raw Online Retail II Excel dataset and export a cleaned subset
# to CSV for downstream analysis and modeling.

import pandas as pd
import os

INPUT_FILE = 'online_retail_II.xlsx'
OUTPUT_FILE = 'cleaned_online_retail.csv'

print(f"Loading data from {INPUT_FILE}...")

# Sanity check to ensure the raw input file is present.
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"The input file {INPUT_FILE} does not exist.")

# Read the specified sheet from the Excel file; report basic shape for reference.
try:
    df = pd.read_excel(INPUT_FILE, sheet_name='Year 2009-2010', engine='openpyxl')
    print(f"Initial data shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()
    
print("Data Preprocessing...")

# Normalize column names by stripping whitespace to avoid key-matching issues.
df.columns = df.columns.str.strip()

# Remove rows missing a Customer ID since they cannot be associated with a customer.
initial_rows = len(df)
df.dropna(subset=['Customer ID'], inplace=True)

print(f"Dropped {initial_rows - len(df)} rows with missing Customer ID.")

# Filter out transactions with non-positive quantities or prices, which typically
# correspond to cancellations or data entry errors.
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
print(f"Data shape after removing negative Quantity and Price: {df.shape}")

# Compute the total monetary value for each line item.
df['TotalAmount'] = df['Quantity'] * df['Price']

# Save the cleaned, enriched dataset to a CSV file for later use.
df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned data saved to {OUTPUT_FILE}")