import streamlit as st
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load environment variables from .env file
load_dotenv()

print(os.getenv('PRODUCTION'))
# Check if running in production
if os.getenv('PRODUCTION') == '1':
    # Use Streamlit secrets for database connection parameters
    db_params = {
        'dbname': st.secrets["DB_NAME"],
        'user': st.secrets["DB_USERNAME"],
        'password': st.secrets["DB_PASSWORD"],
        'host': st.secrets["DB_HOST"],
        'port': st.secrets["DB_PORT"]
    }
elif os.getenv('PRODUCTION') == '0':
    db_params = {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }
else:
    raise ValueError('Invalid value for PRODUCTION environment variable')

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

# Display key metrics
st.title('Sales Dashboard')
st.header('Key Metrics')
col1, col2, col3 = st.columns(3)
col1.metric('Total Sales', f"${total_sales:,.2f}")
col2.metric('Average Sales', f"${average_sales:,.2f}")
col3.metric('Total Orders', total_orders)

# Display top-selling products
st.header('Top Selling Products')
fig = px.bar(top_selling_products, x=top_selling_products.index, y='sales', labels={'x': 'Product Name', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)

# Display sales by region
st.header('Sales by Region')
fig = px.bar(sales_by_region, x=sales_by_region.index, y='sales', labels={'x': 'Region', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)

# Display sales by sub-category
st.header('Sales by Sub-Category')
fig = px.bar(sales_by_sub_category, x=sales_by_sub_category.index, y='sales', labels={'x': 'Sub-Category', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)

# Display sales over time
st.header('Sales Over Time')
fig = px.line(sales_over_time, x=sales_over_time.index, y='sales', labels={'x': 'Order Date', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)

# Display sales by segment
st.header('Sales by Segment')
fig = px.pie(sales_by_segment, values='sales', names=sales_by_segment.index, title='Sales by Segment', template='plotly_dark')
st.plotly_chart(fig)

# Data Mining: Customer Segmentation using K-Means Clustering
st.header('Customer Segmentation')
# Preprocess the data for clustering
data = sales_data.copy()
data['Segment'] = data['segment'].astype('category').cat.codes
data['Region'] = data['region'].astype('category').cat.codes
features = data[['sales', 'Segment', 'Region']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Analyze and visualize the clusters
fig, ax = plt.subplots()
sns.scatterplot(data=data, x='sales', y='Segment', hue='cluster', palette='viridis', ax=ax)
st.pyplot(fig)

# Data Mining: Predictive Analysis using Linear Regression
st.header('Predictive Analysis')
# Preprocess the data for linear regression
data['order_date'] = pd.to_datetime(data['order_date'])
data['Year'] = data['order_date'].dt.year
data['Month'] = data['order_date'].dt.month
data['Day'] = data['order_date'].dt.day

# Using 'Year', 'Month', 'Day' as features and 'sales' as target
X = data[['Year', 'Month', 'Day']]
y = data['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
linear_y_pred = linear_model.predict(X_test)

# Evaluate the model
linear_mse = mean_squared_error(y_test, linear_y_pred)
linear_r2 = r2_score(y_test, linear_y_pred)

st.write(f'Mean Squared Error (Linear Regression): {linear_mse}')
st.write(f'R^2 Score (Linear Regression): {linear_r2}')

# Train a decision tree regression model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
tree_y_pred = tree_model.predict(X_test)

# Evaluate the model
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_r2 = r2_score(y_test, tree_y_pred)

st.write(f'Mean Squared Error (Decision Tree Regression): {tree_mse}')
st.write(f'R^2 Score (Decision Tree Regression): {tree_r2}')

# Train a random forest regression model
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)

# Make predictions
forest_y_pred = forest_model.predict(X_test)

# Evaluate the model
forest_mse = mean_squared_error(y_test, forest_y_pred)
forest_r2 = r2_score(y_test, forest_y_pred)

st.write(f'Mean Squared Error (Random Forest Regression): {forest_mse}')
st.write(f'R^2 Score (Random Forest Regression): {forest_r2}')

# Visualize the actual vs predicted sales for all models
st.header('Actual vs Predicted Sales')
fig, ax = plt.subplots()
plt.scatter(y_test, linear_y_pred, alpha=0.5, label='Linear Regression')
plt.scatter(y_test, tree_y_pred, alpha=0.5, label='Decision Tree Regression', color='red')
plt.scatter(y_test, forest_y_pred, alpha=0.5, label='Random Forest Regression', color='green')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
st.pyplot(fig)

# Feature importance analysis for Random Forest
importances = forest_model.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

# Visualize feature importances
st.header('Feature Importances from Random Forest')
fig, ax = plt.subplots()
forest_importances.sort_values().plot(kind='barh', ax=ax)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(fig)

# Close the database connection
conn.close()