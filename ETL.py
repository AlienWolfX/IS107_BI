import pandas as pd

file_path = 'dataset/train.csv'
data = pd.read_csv(file_path)

# Convert date columns to datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y', errors='coerce')
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d/%m/%Y', errors='coerce')

# Fill missing postal codes with -1 and convert to integer type
data['Postal Code'].fillna(-1, inplace=True)
data['Postal Code'] = data['Postal Code'].astype(int)

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