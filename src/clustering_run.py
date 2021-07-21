'''
   this file contains function read csv
'''
import os
import configparser
from datetime import datetime
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
from src import clustering_read
from src import NeighborsClassifier,Clustering

def run():
    '''
        this function can fetch the dataset file
    '''
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '../config', 'clustering.conf'))
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    neighbor_path = "clustering_result"
    if not os.path.exists(neighbor_path):
        os.mkdir(neighbor_path)
    path=neighbor_path+'/'+time_str
    if not os.path.exists(path):
        os.mkdir(path)
    data= clustering_read.load_data()
    classifier=NeighborsClassifier().model
    parameter_list=[]
    score_list=[]
    target_list=[]
    for n_cluster in list(config['Parameters']['n_clusters'].split(" ")):
        for affinity in list(config['Parameters']['affinity'].split(" ")):
            cluster=Clustering(int(n_cluster),affinity).model
            target=cluster.fit_predict(data)
            classifier.fit(data,target)
            score_list.append(classifier.score(data,target))
            parameter_list.append(f"n_clusters: {n_cluster} affinity: {affinity}")
            target_list.append(target)
    save = pd.DataFrame({'score': score_list,
                         'parameters': parameter_list})
    save.to_csv(path + "/" + time_str + ".csv", index=False)
    analysis=PCA(2)
    features=analysis.fit_transform(data,target_list[0])
    plt.figure(figsize=(10, 4))
    plt.plot(features[target_list[0]==0][0],features[target_list[0]==0][1],"b",label="class 0")
    plt.plot(features[target_list[0] == 1][0], features[target_list[0] == 1][1], "g",label="class 1")
    plt.legend()
    plt.savefig(path + "/" + time_str + ".png")
