import streamlit as st
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables from .env file
load_dotenv()

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
sales_by_segment = sales_data.groupby('segment')['sales'].sum()
sales_over_time = sales_data.groupby('order_date')['sales'].sum()

# Custom color palette and theme
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
chart_template = 'plotly_white'
chart_defaults = dict(
    template=chart_template,
    color_discrete_sequence=custom_colors
)

# Display key metrics
st.title('Sales Dashboard')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Sales', f"${total_sales:,.2f}")
with col2:
    st.metric('Average Sales', f"${average_sales:,.2f}")
with col3:
    st.metric('Total Orders', f"{total_orders:,}")

# Enhanced top-selling products chart
st.subheader('Top Selling Products')
fig = px.bar(top_selling_products, 
             x=top_selling_products.index, 
             y='sales',
             **chart_defaults,
             labels={'x': 'Product Name', 'sales': 'Sales ($)'},
             title='Top 5 Products by Sales')
fig.update_layout(
    showlegend=False,
    xaxis_tickangle=-45,
    hoverlabel=dict(bgcolor="white"),
    title_x=0.5
)
st.plotly_chart(fig, use_container_width=True)

# Enhanced sales by region chart
st.subheader('Sales by Region')
fig = px.bar(sales_by_region,
             x=sales_by_region.index,
             y='sales',
             **chart_defaults,
             labels={'x': 'Region', 'sales': 'Sales ($)'},
             title='Regional Sales Distribution')
fig.update_layout(
    title_x=0.5,
    bargap=0.2
)
st.plotly_chart(fig, use_container_width=True)

# Enhanced sales by sub-category chart
st.subheader('Sales by Sub-Category')
fig = px.bar(sales_by_sub_category.sort_values(ascending=True),
             orientation='h',
             **chart_defaults,
             labels={'x': 'Sales ($)', 'y': 'Sub-Category'},
             title='Sales Performance by Product Sub-Category')
fig.update_layout(
    title_x=0.5,
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Enhanced sales over time chart
st.subheader('Sales Over Time')
fig = px.line(sales_over_time,
              **chart_defaults,
              labels={'x': 'Order Date', 'sales': 'Sales ($)'},
              title='Sales Trend Analysis')
fig.update_layout(
    title_x=0.5,
    xaxis_rangeslider_visible=True
)
st.plotly_chart(fig, use_container_width=True)

# Enhanced sales by segment pie chart
st.subheader('Sales by Segment')
fig = px.pie(sales_by_segment,
             values='sales',
             names=sales_by_segment.index,
             **chart_defaults,
             title='Revenue Distribution by Customer Segment',
             hole=0.4)
fig.update_layout(
    title_x=0.5,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# Close the database connection
conn.close()