'''
   this file contains class classifier
'''
from sklearn.cluster import AgglomerativeClustering


class Clustering:
    '''
        this class is Clustering
    '''
    def __init__(self,n_clusters=5, affinity="euclidean",):
        '''
            this method initialize
        '''
        self.affinity=affinity
        self.n_clusters=n_clusters
        self.model=AgglomerativeClustering\
            (n_clusters=n_clusters\
            ,affinity="euclidean")
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
