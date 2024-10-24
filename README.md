<div style="text-align: center">

# IS107 - Business Intelligence

## Semestral Performance Task

### Building a Business Intelligence solution for a Retail Store

</div>

<br />

<br />

#### Notes

First step is to read the csv file with pandas using read_csv function

```bash
def readCSV(path):
    return pd.read_csv(path)
```

Next is to transform the data using the process

Drop missing values (Destructive?)

```bash
cleanedData = data.dropna()
```

Fill missing values ([Refer here](https://www.geeksforgeeks.org/python-pandas-dataframe-ffill/))

```bash
cleanedData = data.fillna(method='ffill')
```

Standardize text format (e.g., lowercase)

```bash
for column in cleanedData.select_dtypes(include=['object']).columns:
    cleanedData.loc[:, column] = cleanedData[column].str.lower()
```

Save the cleaned dataset

```bash
def saveData(data, output):
    data.to_csv(output, index=False)
```