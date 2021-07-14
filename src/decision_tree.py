'''
   this file contains class Decision tree
'''
from sklearn.tree import DecisionTreeRegressor


class DecisionTree(DecisionTreeRegressor):
    '''
        this class is decision tree
    '''
    def __init__(self, **kwargs):
        '''
           this method initialize
        '''
        super().__init__(**kwargs)
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
