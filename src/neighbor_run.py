'''
   this file contains function read csv
'''
import os
import configparser
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src import neighbor_read
from src import NeighborsClassifier

def run():
    '''
        this function can fetch the dataset file
    '''
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '../config', 'classifier.conf'))
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    neighbor_path = "neighbor_classifier_result"
    if not os.path.exists(neighbor_path):
        os.mkdir(neighbor_path)
    path=neighbor_path+'/'+time_str
    if not os.path.exists(path):
        os.mkdir(path)
    input_train_neighbor, input_test_neighbor, \
    target_train_neighbor, target_test_neighbor = neighbor_read.load_data()
    param_grid = [
        {'n_neighbors': [int(_) \
        for _ in list(config['Parameters']['n_neighbors'].split(" "))],
         'leaf_size': [int(_) \
        for _ in list(config['Parameters']['leaf_size'].split(" "))]}
    ]
    neighbor = NeighborsClassifier().model
    grid_search = GridSearchCV(neighbor, param_grid, cv=5,
                               return_train_score=True)
    grid_search.fit(input_train_neighbor, target_train_neighbor)
    cvres = grid_search.cv_results_
    save = pd.DataFrame({'score': cvres["mean_test_score"],
                         'parameters': cvres["params"]})
    save.to_csv(path + "/" + time_str + ".csv", index=False)
    output_neighbor = grid_search.best_estimator_.predict(input_test_neighbor)
    zero_counter = 0
    one_counter = 0
    plot_list = []
    for target in target_test_neighbor:
        if target == 0:
            zero_counter = zero_counter + 1
        elif target == 1:
            one_counter = one_counter + 1
    plot_list.append(zero_counter)
    plot_list.append(one_counter)
    zero_counter = 0
    one_counter = 0
    for target in output_neighbor:
        if target == 0:
            zero_counter = zero_counter + 1
        elif target == 1:
            one_counter = one_counter + 1
    plot_list.append(zero_counter)
    plot_list.append(one_counter)
    axis = np.arange(1, 5)
    width = 0.2
    _, axis_result = plt.subplots()
    _ = axis_result.bar(axis, plot_list, width)
    axis_result.set_ylabel('Number')
    axis_result.set_xticks(axis)
    axis_result.set_xticklabels(['target_class1', 'target_class2' \
                           , 'output_class1', 'output_class2'])
    plt.savefig(path + "/" + time_str + ".png")
