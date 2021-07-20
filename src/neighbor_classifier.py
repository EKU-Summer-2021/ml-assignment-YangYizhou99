'''
   this file contains class classifier
'''
from sklearn.neighbors import KNeighborsClassifier


class NeighborsClassifier:
    '''
        this class is NeighborsClassifier
    '''
    def __init__(self, n_neighbors=10,leaf_size=30):
        '''
            this method initialize
        '''
        self.n_neighbors=n_neighbors
        self.leaf_size=leaf_size
        self.model=KNeighborsClassifier(n_neighbors=n_neighbors\
                                      ,leaf_size=leaf_size)
        self.nothing=0

    def do_right2(self):
        '''
            this method do nothing
        '''
        return self.nothing

    def do_left2(self):
        '''
            this method do nothing
        '''
        return self.nothing
