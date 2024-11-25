import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Connect to PostgreSQL
conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Create the dimension tables
cur.execute('''
CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id VARCHAR(10) PRIMARY KEY,
    customer_name VARCHAR(100),
    segment VARCHAR(50),
    country VARCHAR(50),
    city VARCHAR(50),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    region VARCHAR(50)
);
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS dim_products (
    product_id VARCHAR(15) PRIMARY KEY,
    category VARCHAR(50),
    sub_category VARCHAR(50),
    product_name VARCHAR(255)
);
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS dim_time (
    date_id DATE PRIMARY KEY,
    year INT,
    month INT,
    day INT,
    quarter INT
);
''')

# Create the fact table
cur.execute('''
CREATE TABLE IF NOT EXISTS fact_sales (
    row_id SERIAL PRIMARY KEY,
    order_id VARCHAR(20),
    order_date DATE,
    ship_date DATE,
    ship_mode VARCHAR(50),
    customer_id VARCHAR(10),
    product_id VARCHAR(15),
    sales NUMERIC,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
    FOREIGN KEY (order_date) REFERENCES dim_time(date_id)
);
''')

# Commit the changes
conn.commit()

# Load data from CSV
df = pd.read_csv('dataset/cleaned_train.csv')

# Insert data into the dimension tables
for _, row in df.iterrows():
    cur.execute('''
    INSERT INTO dim_customers (customer_id, customer_name, segment, country, city, state, postal_code, region)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (customer_id) DO NOTHING;
    ''', (row['Customer ID'], row['Customer Name'], row['Segment'], row['Country'], row['City'], row['State'], row['Postal Code'], row['Region']))

    cur.execute('''
    INSERT INTO dim_products (product_id, category, sub_category, product_name)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (product_id) DO NOTHING;
    ''', (row['Product ID'], row['Category'], row['Sub-Category'], row['Product Name']))

    cur.execute('''
    INSERT INTO dim_time (date_id, year, month, day, quarter)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (date_id) DO NOTHING;
    ''', (row['Order Date'], pd.to_datetime(row['Order Date']).year, pd.to_datetime(row['Order Date']).month, pd.to_datetime(row['Order Date']).day, (pd.to_datetime(row['Order Date']).month - 1) // 3 + 1))

# Insert data into the fact table
for _, row in df.iterrows():
    cur.execute('''
    INSERT INTO fact_sales (order_id, order_date, ship_date, ship_mode, customer_id, product_id, sales)
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    ''', (row['Order ID'], row['Order Date'], row['Ship Date'], row['Ship Mode'], row['Customer ID'], row['Product ID'], row['Sales']))

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()