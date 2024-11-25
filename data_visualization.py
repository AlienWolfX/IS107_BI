import streamlit as st
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import plotly.express as px

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

@st.cache_data
def load_data(query):
    return pd.read_sql(query, conn)

# Load sales data
sales_query = '''
SELECT s.order_id, s.order_date, s.ship_date, s.ship_mode, s.sales, 
       c.customer_name, c.segment, c.country, c.city, c.state, c.region, 
       p.product_name, p.category, p.sub_category
FROM fact_sales s
JOIN dim_customers c ON s.customer_id = c.customer_id
JOIN dim_products p ON s.product_id = p.product_id
'''
sales_data = load_data(sales_query)

# Sidebar filters
st.sidebar.header('Filters')
date_range = st.sidebar.date_input('Date range', [])
category = st.sidebar.multiselect('Product Category', sales_data['category'].unique())
sub_category = st.sidebar.multiselect('Product Sub-Category', sales_data['sub_category'].unique())
region = st.sidebar.multiselect('Region', sales_data['region'].unique())

# Apply filters
if date_range:
    sales_data = sales_data[(sales_data['order_date'] >= date_range[0]) & (sales_data['order_date'] <= date_range[1])]
if category:
    sales_data = sales_data[sales_data['category'].isin(category)]
if sub_category:
    sales_data = sales_data[sales_data['sub_category'].isin(sub_category)]
if region:
    sales_data = sales_data[sales_data['region'].isin(region)]

# Key metrics
total_sales = sales_data['sales'].sum()
average_sales = sales_data['sales'].mean()
total_orders = sales_data['order_id'].nunique()
top_selling_products = sales_data.groupby('product_name')['sales'].sum().nlargest(5)
sales_by_region = sales_data.groupby('region')['sales'].sum()
sales_by_sub_category = sales_data.groupby('sub_category')['sales'].sum()

# Display key metrics
st.title('Sales Dashboard')
st.metric('Total Sales', f"${total_sales:,.2f}")
st.metric('Average Sales', f"${average_sales:,.2f}")
st.metric('Total Orders', total_orders)

# Display top-selling products
st.subheader('Top Selling Products')
fig = px.bar(top_selling_products, x=top_selling_products.index, y='sales', labels={'x': 'Product Name', 'sales': 'Sales'})
st.plotly_chart(fig)

# Display sales by region
st.subheader('Sales by Region')
fig = px.bar(sales_by_region, x=sales_by_region.index, y='sales', labels={'x': 'Region', 'sales': 'Sales'})
st.plotly_chart(fig)

# Display sales by sub-category
st.subheader('Sales by Sub-Category')
fig = px.bar(sales_by_sub_category, x=sales_by_sub_category.index, y='sales', labels={'x': 'Sub-Category', 'sales': 'Sales'})
st.plotly_chart(fig)

# Display sales over time
st.subheader('Sales Over Time')
sales_over_time = sales_data.groupby('order_date')['sales'].sum()
fig = px.line(sales_over_time, x=sales_over_time.index, y='sales', labels={'x': 'Order Date', 'sales': 'Sales'})
st.plotly_chart(fig)

# Display sales by segment
st.subheader('Sales by Segment')
sales_by_segment = sales_data.groupby('segment')['sales'].sum()
fig = px.pie(sales_by_segment, values='sales', names=sales_by_segment.index, title='Sales by Segment')
st.plotly_chart(fig)

# Close the database connection
conn.close()