import unittest
from src import csv_read
from src import DecisionTree
import numpy as np


class DecisionTreeTest(unittest.TestCase):

    def test_cost_function(self):
        input_train, input_test, target_train, target_test = csv_read.load_data()
        model = DecisionTree(max_depth=2, random_state=42)
        model.fit(input_train, target_train)
        self.assertEqual(target_train[0], 79.99)
