import pandas as pd

file_path = 'dataset/train.csv'
data = pd.read_csv(file_path)

# Convert date columns to datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y', errors='coerce')
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d/%m/%Y', errors='coerce')

# This removes the .0 from the postal codes
data['Postal Code'].fillna(-1, inplace=True)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Standardize categorical data by converting strings to lowercase and stripping extra spaces
categorical_columns = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']
for column in categorical_columns:
    data[column] = data[column].str.strip().str.lower()

# Display the cleaned data and info (optional)
print(data.info())
print(data.head())

# Save the cleaned data to a new CSV file if needed
cleaned_file_path = 'dataset/cleaned_train.csv'
data.to_csv(cleaned_file_path, index=False)
