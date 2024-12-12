-- Create the dimension tables
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

CREATE TABLE IF NOT EXISTS dim_products (
    product_id VARCHAR(15) PRIMARY KEY,
    category VARCHAR(50),
    sub_category VARCHAR(50),
    product_name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS dim_time (
    date_id DATE PRIMARY KEY,
    year INT,
    month INT,
    day INT,
    quarter INT
);

-- Create the fact table
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

-- Insert data into the dimension tables
INSERT INTO dim_customers (customer_id, customer_name, segment, country, city, state, postal_code, region)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (customer_id) DO NOTHING;

INSERT INTO dim_products (product_id, category, sub_category, product_name)
VALUES (%s, %s, %s, %s)
ON CONFLICT (product_id) DO NOTHING;

INSERT INTO dim_time (date_id, year, month, day, quarter)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (date_id) DO NOTHING;

-- Insert data into the fact table
INSERT INTO fact_sales (order_id, order_date, ship_date, ship_mode, customer_id, product_id, sales)
VALUES (%s, %s, %s, %s, %s, %s, %s);