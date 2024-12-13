import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os
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
data = load_data(sales_query)

# Preprocess the data for clustering
# Using 'Sales', 'Segment', and 'Region' for clustering
# Convert categorical variables to numerical
data['Segment'] = data['segment'].astype('category').cat.codes
data['Region'] = data['region'].astype('category').cat.codes
features = data[['sales', 'Segment', 'Region']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Create a figure with subplots for cluster analysis
plt.figure(figsize=(15, 10))

# Plot 1: Sales vs Segment
plt.subplot(2, 2, 1)
sns.scatterplot(data=data, x='sales', y='Segment', hue='cluster', palette='viridis')
plt.title('Sales vs Segment by Cluster')

# Plot 2: Sales vs Region
plt.subplot(2, 2, 2)
sns.scatterplot(data=data, x='sales', y='Region', hue='cluster', palette='viridis')
plt.title('Sales vs Region by Cluster')

# Plot 3: Region vs Segment
plt.subplot(2, 2, 3)
sns.scatterplot(data=data, x='Region', y='Segment', hue='cluster', palette='viridis')
plt.title('Region vs Segment by Cluster')

# Plot 4: Cluster distribution
plt.subplot(2, 2, 4)
sns.countplot(data=data, x='cluster', palette='viridis')
plt.title('Distribution of Clusters')

plt.tight_layout()
plt.show()

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
data['pca1'] = pca_features[:, 0]
data['pca2'] = pca_features[:, 1]

# Visualize the PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='viridis')
plt.title('PCA of Customer Segmentation')
plt.xlabel('PCA X axis')
plt.ylabel('PCA Y axis')
plt.show()

# Preprocess the data for regression
# Convert 'Order Date' to datetime and extract features like year, month, day
data['order_date'] = pd.to_datetime(data['order_date'])
data['Year'] = data['order_date'].dt.year
data['Month'] = data['order_date'].dt.month
data['Day'] = data['order_date'].dt.day

# Using 'Year', 'Month', 'Day' as features and 'Sales' as target
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

print(f'Mean Squared Error (Linear Regression): {linear_mse}')
print(f'R^2 Score (Linear Regression): {linear_r2}')

# Train a decision tree regression model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
tree_y_pred = tree_model.predict(X_test)

# Evaluate the model
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_r2 = r2_score(y_test, tree_y_pred)

print(f'Mean Squared Error (Decision Tree Regression): {tree_mse}')
print(f'R^2 Score (Decision Tree Regression): {tree_r2}')

# Train a random forest regression model
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)

# Make predictions
forest_y_pred = forest_model.predict(X_test)

# Evaluate the model
forest_mse = mean_squared_error(y_test, forest_y_pred)
forest_r2 = r2_score(y_test, forest_y_pred)

print(f'Mean Squared Error (Random Forest Regression): {forest_mse}')
print(f'R^2 Score (Random Forest Regression): {forest_r2}')

# Visualize the actual vs predicted sales for all models
plt.figure(figsize=(10, 6))
plt.scatter(y_test, linear_y_pred, alpha=0.5, label='Linear Regression')
plt.scatter(y_test, tree_y_pred, alpha=0.5, label='Decision Tree Regression', color='red')
plt.scatter(y_test, forest_y_pred, alpha=0.5, label='Random Forest Regression', color='green')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()

# Feature importance analysis for Random Forest
importances = forest_model.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

# Visualize feature importances
plt.figure(figsize=(10, 6))
forest_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Close the database connection
conn.close()