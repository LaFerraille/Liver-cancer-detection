# random_baseline.py
import numpy as np

def compute_random_baseline(y):
    """
    Computes the expected random accuracy based on the distribution of classes in the target array.

    Parameters:
    - y (numpy array): Target variable array containing class labels as integers.

    Returns:
    - float: Expected random accuracy.
    """
    unique, counts = np.unique(y, return_counts=True)
    total_samples = y.size
    class_probabilities = counts / total_samples
    random_accuracy = np.sum(class_probabilities ** 2)
    return random_accuracy
