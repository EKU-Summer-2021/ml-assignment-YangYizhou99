import numpy as np
import pandas as pd
from src import tree_read,neighbor_read
from src import DecisionTree,NeighborsClassifier
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import os
from datetime import datetime
if __name__ == '__main__':
    input_train_tree, input_test_tree, target_train_tree,\
    target_test_tree = tree_read.load_data()
    param_grid = [
        {'min_samples_split': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,18,20,21,22,23,24,25,26,27\
                               ], 'min_samples_leaf': [2]}
    ]
    decision_tree = DecisionTree(random_state=42)
    grid_search = GridSearchCV(decision_tree, param_grid, cv=5,
                               return_train_score=True)
    grid_search.fit(input_train_tree, target_train_tree)
    cvres = grid_search.cv_results_
    save = pd.DataFrame({'score': cvres["mean_test_score"],
                         'parameters':cvres["params"]})
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    decision_tree_path = "decision_tree_result"
    if not os.path.exists(decision_tree_path):
        os.mkdir(decision_tree_path)
    save.to_csv(decision_tree_path+"/"+time_str+".csv", index=False)
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(grid_search.best_estimator_,
                       max_depth=3,
                       feature_names=["cement", "slag", "flyash", "water", "superplasticizer", "coarseaggregate",
                                      "fineaggregate", "age"],
                       filled=True)

    fig.savefig(decision_tree_path+"/"+time_str+".png")
    # new line
    # new line
    neighbor_path="neighbor_classifier_result"
    if not os.path.exists(neighbor_path):
        os.mkdir(neighbor_path)
    input_train_neighbor, input_test_neighbor, \
    target_train_neighbor, target_test_neighbor = neighbor_read.load_data()
    param_grid = [
        {'n_neighbors': [15,20,30,31,32,33,34,35,36,32,39], 'leaf_size': [5,6,4,7,8,9]}
    ]
    neighbor = NeighborsClassifier().model
    grid_search = GridSearchCV(neighbor , param_grid, cv=5,
                               return_train_score=True)
    grid_search.fit(input_train_neighbor, target_train_neighbor)
    cvres = grid_search.cv_results_
    save = pd.DataFrame({'score': cvres["mean_test_score"],
                         'parameters': cvres["params"]})
    save.to_csv(neighbor_path+"/"+time_str+".csv", index=False)
    output_neighbor=grid_search.best_estimator_.predict(input_test_neighbor)
    zero_counter=0
    one_counter=0
    plot_list=[]
    for target in target_test_neighbor:
        if target==0:
            zero_counter=zero_counter+1
        elif target==1:
            one_counter=one_counter+1
    plot_list.append(zero_counter)
    plot_list.append(one_counter)
    zero_counter = 0
    one_counter = 0
    for target in output_neighbor:
        if target==0:
            zero_counter=zero_counter+1
        elif target==1:
            one_counter=one_counter+1
    plot_list.append(zero_counter)
    plot_list.append(one_counter)
    axis = np.arange(1, 5)
    width = 0.2
    fig, ax = plt.subplots()
    _ = ax.bar(axis, plot_list, width)
    ax.set_ylabel('Number')
    ax.set_xticks(axis)
    ax.set_xticklabels(['target_class1','target_class2'\
                        ,'output_class1','output_class2'])
    plt.savefig(neighbor_path+"/"+time_str+".png")





