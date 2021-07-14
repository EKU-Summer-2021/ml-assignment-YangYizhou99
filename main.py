import numpy as np
import pandas as pd
from src import csv_read
from src import DecisionTree
from matplotlib import pyplot as plt
from sklearn import tree
if __name__ == '__main__':
    input_train, input_test, target_train, target_test = csv_read.load_data()
    model=DecisionTree(max_depth=2, random_state=42)
    model.fit(input_train,target_train)
    output=model.predict(input_test)
    save = pd.DataFrame({'output': output})
    save.to_csv('result.csv', index=False)
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model)
    fig.savefig("decistion_tree.png")
