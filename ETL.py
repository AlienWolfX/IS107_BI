import pandas as pd

def readCSV(path):
    return pd.read_csv(path)

def transformData(data):
    cleanedData = data.dropna()
    # cleanedData = data.fillna(method='ffill')  # using ffill 'https://www.geeksforgeeks.org/python-pandas-dataframe-ffill/'

    for column in cleanedData.select_dtypes(include=['object']).columns:
        cleanedData.loc[:, column] = cleanedData[column].str.lower()

    return cleanedData

def saveData(data, output):
    data.to_csv(output, index=False)

path = 'dataset/train.csv'
output = 'dataset/train(cleaned).csv'

csv = readCSV(path)

transform = transformData(csv)

saveData(transform, output)

print(transform.head())