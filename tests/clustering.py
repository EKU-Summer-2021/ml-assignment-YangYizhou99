import unittest
from src import clustering_read
from src import Clustering
from src import clustering_run
import os

class NeighborsClassifierTest(unittest.TestCase):

    def test_cost_function(self):
        data= clustering_read.load_data()
        model = Clustering().model
        model.fit_predict(data)
        clustering_run.run()
        self.assertTrue(os.path.exists('clustering_result'))