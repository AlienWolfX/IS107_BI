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

# Display key metrics with insights
st.title('Sales Dashboard')
st.header('Key Metrics')
col1, col2, col3 = st.columns(3)
col1.metric('Total Sales', f"${total_sales:,.2f}")
col2.metric('Average Sales', f"${average_sales:,.2f}")
col3.metric('Total Orders', total_orders)

# Display top-selling products with insights
st.header('Top Selling Products')
fig = px.bar(top_selling_products, x=top_selling_products.index, y='sales', labels={'x': 'Product Name', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)
st.markdown(f"""
*Key Insights:*
- *Top performing product: {top_selling_products.index[0]} (${top_selling_products.iloc[0]:,.2f} in sales)*
""")

# Display sales by region with insights
st.header('Sales by Region')
fig = px.bar(sales_by_region, x=sales_by_region.index, y='sales', labels={'x': 'Region', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)
st.markdown(f"""
*Key Insights:*
- *Highest performing region: {sales_by_region.idxmax()} (${sales_by_region.max():,.2f})*
- *Lowest performing region: {sales_by_region.idxmin()} (${sales_by_region.min():,.2f})*
- *Regional performance variance: {(sales_by_region.max() - sales_by_region.min())/sales_by_region.mean()*100:.1f}%*
""")

# Display sales by sub-category with insights
st.header('Sales by Sub-Category')
fig = px.bar(sales_by_sub_category, x=sales_by_sub_category.index, y='sales', labels={'x': 'Sub-Category', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)
st.markdown(f"""
*Key Insights:*
- *Top sub-category: {sales_by_sub_category.idxmax()} (${sales_by_sub_category.max():,.2f})*
- *Bottom sub-category: {sales_by_sub_category.idxmin()} (${sales_by_sub_category.min():,.2f})*
- *Number of sub-categories: {len(sales_by_sub_category)}*
""")

# Display sales over time with insights
st.header('Sales Over Time')
fig = px.line(sales_over_time, x=sales_over_time.index, y='sales', labels={'x': 'Order Date', 'sales': 'Sales'}, template='plotly_dark')
st.plotly_chart(fig)
st.markdown(f"""
*Key Insights:*
- *Peak sales day: {sales_over_time.idxmax().strftime('%Y-%m-%d')} (${sales_over_time.max():,.2f})*
- *Average daily sales: ${sales_over_time.mean():,.2f}*
- *Sales trend indicates {'upward' if sales_over_time.iloc[-1] > sales_over_time.iloc[0] else 'downward'} movement*
""")

# Display sales by segment with insights
st.header('Sales by Segment')
fig = px.pie(sales_by_segment, values='sales', names=sales_by_segment.index, title='Sales by Segment', template='plotly_dark')
st.plotly_chart(fig)
st.markdown(f"""
*Key Insights:*
- *Dominant segment: {sales_by_segment.idxmax()} ({(sales_by_segment.max()/sales_by_segment.sum()*100):.1f}% of total sales)*
- *Smallest segment: {sales_by_segment.idxmin()} ({(sales_by_segment.min()/sales_by_segment.sum()*100):.1f}% of total sales)*
- *Number of customer segments: {len(sales_by_segment)}*
""")

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

# Add insights for Customer Segmentation
st.markdown("""
*Clustering Insights:*
- *Customers grouped into 3 distinct clusters based on sales, segment, and region*
- *Clusters show different purchasing patterns and regional preferences*
""")

# Fix cluster insights aggregation
cluster_insights = data.groupby('cluster').agg({
    'sales': ['mean', 'count', 'std'],
    'Segment': lambda x: data.loc[x.index, 'segment'].mode()[0],  
    'Region': lambda x: data.loc[x.index, 'region'].mode()[0]    
}).round(2)

st.markdown("""
## Customer Segmentation Insights

### Cluster Analysis Summary:
""")

# Update cluster analysis display
for cluster in range(3):
    avg_sales = cluster_insights['sales']['mean'][cluster]
    count = cluster_insights['sales']['count'][cluster]
    try:
        segment = cluster_insights['Segment'][cluster]
        region = cluster_insights['Region'][cluster]
    except:
        segment = "Not available"
        region = "Not available"
    
    st.markdown(f"""
    #### Cluster {cluster}:
    - **Size**: {count} customers ({count/len(data)*100:.1f}% of total)
    - **Average Sales**: ${avg_sales:,.2f}
    - **Primary Segment**: {segment}
    - **Main Region**: {region}
    - **Sales Variation**: ${cluster_insights['sales']['std'][cluster]:,.2f}
    """)

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

# Add insights for Predictive Analysis
st.markdown(f"""
*Model Performance Insights:*
- *Best performing model: {
    '**Linear Regression**' if linear_r2 > tree_r2 and linear_r2 > forest_r2 
    else '**Decision Tree**' if tree_r2 > forest_r2 
    else '**Random Forest**'
}*
- *Model accuracy comparison:*
  * *Linear Regression: {linear_r2:.2%}*
  * *Decision Tree: {tree_r2:.2%}*
  * *Random Forest: {forest_r2:.2%}*
""")

# Add after the predictive analysis section
st.markdown("""
## Sales Prediction Insights

### Model Performance Analysis:
""")

models_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'R² Score': [linear_r2, tree_r2, forest_r2],
    'MSE': [linear_mse, tree_mse, forest_mse]
})

best_model = models_comparison.loc[models_comparison['R² Score'].idxmax()]
st.markdown(f"""
#### Key Findings:
- **Best Performing Model**: {best_model['Model']}
- **Prediction Accuracy**: {best_model['R² Score']:.2%}
- **Error Metric (MSE)**: {best_model['MSE']:.2f}

#### Time-based Patterns:
- **Yearly Trend**: {'Increasing' if linear_model.coef_[0] > 0 else 'Decreasing'}
- **Seasonal Patterns**: Most sales occur in {'Q4' if data.groupby('Month')['sales'].mean().idxmax() > 9 else 'Q2'}
""")

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

# Add after feature importance visualization
st.markdown("""
## Business Recommendations

### Based on Analysis:
1. **Customer Strategy**:
   - Focus on high-value cluster for retention
   - Develop targeted promotions for each segment
   - Geographic expansion opportunities identified

2. **Sales Optimization**:
   - Leverage predictive models for inventory planning
   - Focus on top-performing product categories
   - Adjust regional strategies based on performance

3. **Future Planning**:
   - Implement {best_model['Model']} for sales forecasting
   - Monitor seasonal patterns for stock management
   - Develop region-specific growth strategies
""")

# Close the database connection
conn.close()