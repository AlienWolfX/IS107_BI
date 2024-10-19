import pandas as pd

"""
Functions below read the data from the given path

readExcel(path) - reads the data from the excel file
readCsv(path) - reads the data from the csv file    
"""

def readExcel(path):
    excel = pd.read_excel(path)
    print(excel.head())

def readCsv(path):
    csv = pd.read_csv(path)
    print(csv.head())
    

x = input("Choose an Option: ")

match x:
    case '1':
        readCsv('archive/train.csv')
    case '2':
        readExcel('onlineRetail/onlineRetail.xlsx')
    case _:
        print("Invalid Option")

# readCsv('archive/train.csv')
# readExcel('onlineRetail/onlineRetail.xlsx')