'''
Example module for template project.
Pylint will check code in the src directory only!
'''
from src.decision_tree import DecisionTree
from src.neighbor_classifier import NeighborsClassifier
from src.clustering import Clustering
__all__ = [
    'DecisionTree',
    'NeighborsClassifier',
    'Clustering'
]
