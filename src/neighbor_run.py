'''
   this file contains function read csv
'''
import os
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
        {'n_neighbors': [15, 20, 30, 31, 32, 33, 34, \
                         35, 36, 32, 39], \
         'leaf_size': [5, 6, 4, 7, 8, 9]}
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
    _, axis = plt.subplots()
    _ = axis.bar(axis, plot_list, width)
    axis.set_ylabel('Number')
    axis.set_xticks(axis)
    axis.set_xticklabels(['target_class1', 'target_class2' \
                           , 'output_class1', 'output_class2'])
    plt.savefig(path + "/" + time_str + ".png")
