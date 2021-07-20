'''
   this file contains function read csv
'''
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

def fetch_data():
    '''
        this function can fetch the dataset file
    '''
    path = "https://dayangai.c"\
    "oding.net/p/dayangai/d/daya"\
    "ngai/git/raw/master/dataset"\
    "s/Concrete_Data_Yeh.cs"\
    "v?download=true"
    urllib.request.urlretrieve(path, "Concrete_Data_Yeh.csv")

def load_data():
    '''
        this function can load csv file data
    '''
    fetch_data()
    csv_path = "Concrete_Data_Yeh.csv"
    dataset=pd.read_csv(csv_path)
    data = dataset.drop("csMPa", axis=1)
    target = dataset.csMPa.copy()

    input_train, input_test, target_train, target_test = train_test_split(data, target, test_size=0.15,random_state=42)
    return input_train, input_test, target_train, target_test
