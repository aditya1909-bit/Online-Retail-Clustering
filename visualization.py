#Aditya Dutta & Ege Gürsel

# Generate descriptive visualizations for product performance and temporal
# shopping patterns from the cleaned online retail dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = 'cleaned_online_retail.csv'

print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Ensure invoice dates are in datetime format for time-based analysis.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# If TotalAmount is not already present, compute revenue per line item.
if 'TotalAmount' not in df.columns:
    df['TotalAmount'] = df['Quantity'] * df['Price']

sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (12, 6)})

print("Generating Top 10 Products by Quantity plot...")

# Compute the total quantity sold for each product and select the top 10.
top_qty = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_qty.values, y=top_qty.index, palette='viridis')
plt.title('Top 10 Best-Selling Products (By Quantity Sold)')
plt.xlabel('Total Units Sold')
plt.ylabel('Product Name')
plt.tight_layout()
plt.savefig('viz_top_quantity.png')
print("Saved 'viz_top_quantity.png'")

print("Generating Top 10 Products by Revenue plot...")

# Compute total revenue per product and select the top 10 highest-earning items.
top_rev = df.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_rev.values, y=top_rev.index, palette='magma')
plt.title('Top 10 High-Value Products (By Total Revenue)')
plt.xlabel('Total Revenue (£)')
plt.ylabel('Product Name')
plt.tight_layout()
plt.savefig('viz_top_revenue.png')
print("Saved 'viz_top_revenue.png'")

print("Generating Hourly Sales Traffic plot...")

# Extract the hour of day from each invoice timestamp to analyze intra-day patterns.
df['Hour'] = df['InvoiceDate'].dt.hour

# Count the number of unique orders per hour to approximate website traffic.
hourly_sales = df.groupby('Hour')['Invoice'].nunique()

plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_sales.index, y=hourly_sales.values, marker='o', linewidth=2.5, color='dodgerblue')
plt.title('Website Traffic: Number of Orders per Hour')
plt.xlabel('Hour of Day (24h)')
plt.ylabel('Number of Unique Orders')
plt.xticks(range(6, 22))
plt.grid(True)
plt.tight_layout()
plt.savefig('viz_hourly_traffic.png')
print("Saved 'viz_hourly_traffic.png'")

print("\nDone! Check your folder for the 3 PNG files.")