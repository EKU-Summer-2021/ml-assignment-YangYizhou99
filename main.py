import numpy as np
import pandas as pd
from src import csv_read
from src import DecisionTree
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import GridSearchCV
if __name__ == '__main__':
    input_train, input_test, target_train, target_test = csv_read.load_data()
    param_grid = [
        {'min_samples_split': [3, 10, 30], 'min_samples_leaf': [2, 4, 6, 8]}
    ]
    model = DecisionTree(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5,
                               return_train_score=True)
    grid_search.fit(input_train, target_train)
    cvres = grid_search.cv_results_
    # model.fit(input_train,target_train)
    # output=model.predict(input_test)
    save = pd.DataFrame({'score': cvres["mean_test_score"],
                         'parameters':cvres["params"]})
    save.to_csv('result.csv', index=False)
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(grid_search.best_estimator_)
    fig.savefig("decistion_tree.png")
