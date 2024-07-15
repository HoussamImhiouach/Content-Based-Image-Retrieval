# distances.py
import numpy as np
from scipy.spatial import distance

def manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2))

def euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def chebyshev(v1, v2):
    return np.max(np.abs(v1 - v2))

def canberra(v1, v2):
    return distance.canberra(v1, v2)

def retrieve_similar_images(signature_db, query_features, distance_func, num_results):
    distances = []
    for instance in signature_db:
        label, img_path, features = instance[-2], instance[-1], instance[:-2]
        dist = distance_func(query_features, features)
        distances.append((img_path, dist, label))
    
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]
