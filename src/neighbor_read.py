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
    path = "https://storage.googleapis.com/kagg"\
           "lesdsdata/datasets/33180/43520/heart.c"\
            "sv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-"\
            "Goog-Credential=gcp-kaggle-com%40kaggle"\
           "-161607.iam.gserviceaccount.com%2F202107"\
           "14%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-"\
            "Date=20210714T104535Z&X-Goog-Expires=259199"\
           "&X-Goog-SignedHeaders=host&X-Goog-Signature=94"\
            "10c88b139be62705d595cd2645493ade0ab434610689e0"\
            "27de4698a327b34d5c0583cfde903d980347ac6b2f614c1"\
            "cce45b2861a4638bd436edf1670411c97b5018f890381115e"\
           "8f078ed94bfcc7efdcb3d3dd59df763c34ffaceff7df1acebf"\
            "99b28699dddb7963cd51496852afb426800a1e6789b9159c105"\
            "aaf28b85b837b22e21b59bed88648f36dc61984c0c34d9003786"\
            "766de52599a3f1536ab8cdb852043dbbeac86960c8f628b3182c2"\
            "138923e75303acdb3757122de939287c637d8630b41042390e015"\
            "1ec54f70c174f87b784fdd4c357ac43fe0e4a0b1a70f9db117b3d48"\
            "ae8ca6d9987d2c4370b5fb82b6d889649cc64e7cb9af23e880c8ae"
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
