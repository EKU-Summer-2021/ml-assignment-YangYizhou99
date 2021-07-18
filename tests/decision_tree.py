import unittest
from src import tree_read
from src import DecisionTree
from src import tree_run
import os


class DecisionTreeTest(unittest.TestCase):

    def test_function(self):
        input_train, input_test, target_train, target_test = tree_read.load_data()
        model = DecisionTree(max_depth=2, random_state=42)
        model.fit(input_train, target_train)
        output=model.predict(input_test)
        tree_run.run()
        self.assertEqual(output[0], 36.68636761487964)
        self.assertTrue(os.path.exists('decision_tree_result'))


