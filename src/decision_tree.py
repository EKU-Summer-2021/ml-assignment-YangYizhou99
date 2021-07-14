'''
   this file contains class Decision tree
'''
from sklearn.tree import DecisionTreeRegressor


class DecisionTree(DecisionTreeRegressor):
    '''
        this class is decision tree
    '''
    def __init__(self, *,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 ccp_alpha=0.0):
        '''
           this method initialize
        '''
        super().__init__(criterion=criterion,
                 splitter=splitter,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 random_state=random_state,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 ccp_alpha=ccp_alpha)
        self.nothing=0

    def do_right(self):
        '''
            this method do nothing
        '''
        return self.nothing

    def do_left(self):
        '''
            this method do nothing
        '''
        return self.nothing
