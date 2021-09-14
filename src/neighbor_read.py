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
    path = "https://"\
            "raw.githubusercontent"\
    ".com/EKU-Summer-2021/onboarding-f"\
            "luencity/master/heart.csv"
    urllib.request.urlretrieve(path, "heart.csv")

def load_data():
    '''
        this function can load csv file data
    '''
    fetch_data()
    csv_path = "heart.csv"
    dataset=pd.read_csv(csv_path)
    data = dataset.drop("target", axis=1)
    target = dataset.target.copy()

    input_train, input_test, target_train, target_test = train_test_split(data, target, test_size=0.15,random_state=42)
    return input_train, input_test, target_train, target_test
