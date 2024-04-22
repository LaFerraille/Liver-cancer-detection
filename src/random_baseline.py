# random_baseline.py
import numpy as np

def compute_random_baseline(y):
   
    unique, counts = np.unique(y, return_counts=True)
    total_samples = y.size
    class_probabilities = counts / total_samples
    random_accuracy = np.sum(class_probabilities ** 2)
    return random_accuracy
