'''
   this file contains function read csv
'''
import os
from datetime import datetime
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pandas as pd
from matplotlib import pyplot as plt
from src import tree_read
from src import DecisionTree


def run():
    '''
        this function can fetch the dataset file
    '''
    input_train_tree, _, target_train_tree, \
    _ = tree_read.load_data()
    param_grid = [
        {'min_samples_split': [3, 4, 5, 6,\
                               7, 8, 9, 10,\
                               11, 12, 13, 14, \
                               15, 16, 17, 19, 18,\
                               20, 21, 22, 23, 24, 25, 26,
                               27 \
                               ], 'min_samples_leaf': [2]}
    ]
    decision_tree = DecisionTree(random_state=42)
    grid_search = GridSearchCV(decision_tree, param_grid, cv=5,
                               return_train_score=True)
    grid_search.fit(input_train_tree, target_train_tree)
    cvres = grid_search.cv_results_
    save = pd.DataFrame({'score': cvres["mean_test_score"],
                         'parameters': cvres["params"]})
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    decision_tree_path = "decision_tree_result"
    if not os.path.exists(decision_tree_path):
        os.mkdir(decision_tree_path)
    path=decision_tree_path+'/'+time_str
    if not os.path.exists(path):
        os.mkdir(path)
    save.to_csv(path + "/" + time_str + ".csv", index=False)
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(grid_search.best_estimator_,
                       max_depth=3,
                       feature_names=["cement", "slag", "flyash",\
                                      "water", "superplasticizer", "coarseaggregate",
                                      "fineaggregate", "age"],
                       filled=True)

    fig.savefig(path + "/" + time_str + ".png")
