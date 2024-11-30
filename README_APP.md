# User Guide for the Application

This guide provides a step-by-step walkthrough of how to use our application which is built with Streamlit.

---

## Overview
The application provides:
1. **Sales Analysis**: Interactive dashboards displaying key sales metrics and visualizations.
2. **Customer Segmentation**: Insights into customer clusters using K-Means clustering.
3. **Predictive Analysis**: Predictions of sales trends using machine learning models.

---

## Getting Started

### Prerequisites
1. Configure your `.env` file with the following variables:
   - `DB_NAME=`
   - `DB_USER=`
   - `DB_PASSWORD=`
   - `DB_HOST=`
   - `DB_PORT=`
2. Install the necessary Python libraries:
   `pip install -r requirements.txt`
3. Run the application using:
   `streamlit run app.py`

---

## Using the Application

### 1. **Sidebar Filters**
   - Navigate to the sidebar to filter the data by:
     - **Date Range**: Select a range of dates for the analysis.
     - **Product Category**: Choose one or more categories.
     - **Product Sub-Category**: Filter by sub-categories.
     - **Region**: Select specific regions.

### 2. **Key Metrics Dashboard**
   - Displays:
     - **Total Sales**: Total revenue generated.
     - **Average Sales**: Average sales value.
     - **Total Orders**: Count of unique orders.

### 3. **Visualizations**
   - **Top Selling Products**: Bar chart of the top 5 products by sales.
   - **Sales by Region**: Regional sales distribution.
   - **Sales by Sub-Category**: Bar chart showing sales distribution across sub-categories.
   - **Sales Over Time**: Line chart of daily sales.
   - **Sales by Segment**: Pie chart of sales distribution by customer segment.

---

## Advanced Features

### 1. **Customer Segmentation**
   - Uses K-Means clustering to group customers based on:
     - Sales
     - Customer Segment
     - Region
   - Visualizes clusters in a scatter plot.

### 2. **Predictive Analysis**
   - Predicts future sales using machine learning models:
     - **Linear Regression**
     - **Decision Tree Regression**
     - **Random Forest Regression**
   - Metrics displayed for each model:
     - Mean Squared Error (MSE)
     - R² Score
   - Visual comparison of actual vs predicted sales values.

### 3. **Feature Importance**
   - Shows the most influential features for predictions using a bar chart from the Random Forest model.

---
