import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv('.env')

db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# URL-encode the password
db_password = urllib.parse.quote_plus(db_password)

print(f"DB_USERNAME: {db_username}")
print(f"DB_PASSWORD: {db_password}")
print(f"DB_HOST: {db_host}")
print(f"DB_PORT: {db_port}")
print(f"DB_NAME: {db_name}")

# Load cleaned data
file_path = 'dataset/cleaned_train.csv'
data = pd.read_csv(file_path)

# Create a connection to PostgreSQL
engine = create_engine(f'postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')

# Load data into fact_sales table
fact_sales = data[['Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID', 'Product ID', 'Sales']]
fact_sales.columns = ['order_id', 'order_date', 'ship_date', 'ship_mode', 'customer_id', 'product_id', 'sales_amount']
fact_sales.to_sql('fact_sales', engine, if_exists='replace', index=False)

# Extract and load data into dimension tables
dim_products = data[['Product ID', 'Category', 'Sub-Category', 'Product Name']].drop_duplicates().reset_index(drop=True)
dim_products.columns = ['product_id', 'category', 'sub_category', 'product_name']
dim_products.to_sql('dim_products', engine, if_exists='replace', index=False)

dim_customers = data[['Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region']].drop_duplicates().reset_index(drop=True)
dim_customers.columns = ['customer_id', 'customer_name', 'segment', 'country', 'city', 'state', 'postal_code', 'region']
dim_customers.to_sql('dim_customers', engine, if_exists='replace', index=False)

# Create a time dimension table
time_data = pd.DataFrame({
    'date': pd.date_range(start=data['Order Date'].min(), end=data['Order Date'].max())
})
time_data['day'] = time_data['date'].dt.day
time_data['month'] = time_data['date'].dt.month
time_data['year'] = time_data['date'].dt.year
time_data['quarter'] = time_data['date'].dt.quarter
time_data.to_sql('dim_time', engine, if_exists='replace', index=False)
