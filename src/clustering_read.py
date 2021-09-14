'''
   this file contains function read csv
'''
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def fetch_data():
    '''
        this function can fetch the dataset file
    '''
    path = "https://raw.githubusercontent.com/EKU-Summer-2021/onboarding-fluencity/master/vgsales.csv"
    urllib.request.urlretrieve(path, "vgsales.csv")

def load_data():
    '''
        this function can load csv file data
    '''
    fetch_data()
    dataset=pd.read_csv("vgsales.csv")
    data = dataset.drop("Name", axis=1)
    data=data.dropna(subset=["Year"])
    data=data.dropna(subset=["Publisher"])
    data_cat = data[["Platform","Genre","Publisher"]]
    data_num = data.drop(["Platform","Genre","Publisher"], \
                         axis=1).to_numpy()
    cat_encoder = OneHotEncoder(sparse=False)
    data_cat = cat_encoder.fit_transform(data_cat)
    data=np.concatenate((data_num, data_cat), axis=1)
    return data[:1000]
