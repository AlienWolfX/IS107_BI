# ETL Documentation

### Transformation Steps and Cleaning Process

1. **Loading the Dataset**:
   - The dataset is loaded from the file `dataset/train.csv` using `pd.read_csv`.

2. **Date Conversion**:
   - The `Order Date` and `Ship Date` columns are converted to datetime format using `pd.to_datetime` with the format specified as `%d/%m/%Y`. The `errors='coerce'` parameter ensures that any invalid date formats are converted to `NaT`.

3. **Handling Missing Postal Codes**:
   - Missing values in the `Postal Code` column are filled with `-1` using `fillna(-1)`.
   - The `Postal Code` column is then converted to integer type using `astype(int)`.

4. **Removing Duplicate Rows**:
   - Duplicate rows in the dataset are removed using `drop_duplicates`.

5. **Standardizing Categorical Data**:
   - Categorical columns (`Ship Mode`, `Segment`, `Country`, `City`, `State`, `Region`, `Category`, `Sub-Category`) are standardized by converting all string values to lowercase and stripping any extra spaces using `str.strip().str.lower()`.

6. **Handling Missing Numerical Values**:
   - Missing values in numerical columns (`Sales`) are filled with the median value of the respective column using `fillna(data[column].median())`.

7. **Outlier Detection and Handling**:
   - Outliers in the `Sales` column are detected and handled using the Interquartile Range (IQR) method:
     - Calculate the first quartile (Q1) and third quartile (Q3) of the `Sales` column.
     - Compute the IQR as `Q3 - Q1`.
     - Define the lower bound as `Q1 - 1.5 * IQR` and the upper bound as `Q3 + 1.5 * IQR`.
     - Filter the dataset to include only rows where `Sales` values are within the lower and upper bounds.

8. **Saving the Cleaned Data**:
   - The cleaned dataset is saved to a new CSV file `dataset/cleaned_train.csv` using `data.to_csv`.

### Challenges Faced

1. **Date Format Variations**:
   - Invalid date formats.

2. **Handling Missing Values**:
   - Missing postal codes with `-1`

3. **Outliers**:
   - Outliers in the `Sales` column.

4. **Standardizing Categorical Data**:
   - Inconsistency in categorical data. 
