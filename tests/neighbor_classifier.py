import unittest
from src import neighbor_read
from src import NeighborsClassifier
import numpy as np


class NeighborsClassifierTest(unittest.TestCase):

    def test_cost_function(self):
        input_train, input_test, target_train, target_test = neighbor_read.load_data()
        model = NeighborsClassifier().model
        model.fit(input_train, target_train)
        output=model.predict(input_test)
        self.assertEqual(output[0], 0)