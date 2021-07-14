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
    path = "https://storage.googleapis.com/kagglesdsdata/"\
           "datasets/31874/41246/Concrete_Data_Yeh.csv?X-Goog"\
           "-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-"\
           "kaggle-com%40kaggle-161607.iam.gserviceaccount.com%"\
           "2F20210714%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Da"\
           "te=20210714T063239Z&X-Goog-Expires=259199&X-Goog-Signed"\
           "Headers=host&X-Goog-Signature=7ab47725fcd63da0b87480d76f"\
            "bcc8afa72b84ec5fbda6b3ca7985f51f19033914f4edd876208f922a4"\
           "8d1cb6335f374aba5360635d9cd624e53b1ebcf39e17050e660c2e9b2b"\
            "a967ffba39c00f1184f8ca3b57707cf00ff4753b4ccd0482e323580e105"\
           "2a61a708bf4ccc5e72dbacff69e43acae8b8d03fcf4929b241b6ac958ae6bc"\
           "402456b03f1549174e83dc64ae6b495f650f2351ce325836929ed16ca3a5"\
           "b409e326f81eb426a1711d12fdfe5537e0fd59fdea9909dd706827c16cac840"\
           "945f9786f2701552f3e55515989e3262987bf42599c14a022a229a8a5279a42d"\
           "8edfb2cbf3eb09e9600c6bccffc1aea1c36337c8b876ef0548aa2d4dc8701e2"
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
