import numpy as np
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
import random

import warnings
warnings.filterwarnings('ignore')

def spectral_clustering_sampling(all_samples, n_clusters):

    n_neighbors=30
    
    # Inicializa o Spectral Clustering.
    # 'affinity=nearest_neighbors' instrui o scikit-learn a construir a matriz de
    # similaridade W usando um grafo de KNN, como descrito no artigo.
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42  # Para reprodutibilidade
    )
    
    # Aplica o agrupamento e obtém os rótulos de cluster
    labels = sc.fit_predict(all_samples)
    
    selected_samples = []
    
    # Para cada cluster, seleciona a amostra mais próxima da média
    for i in range(n_clusters):
        # Encontra os índices das amostras neste cluster
        cluster_indices = np.where(labels == i)[0]
        cluster_data = all_samples[cluster_indices]
        
        # Calcula o centroide (média) do cluster
        cluster_mean = np.mean(cluster_data, axis=0)
        
        # Encontra a amostra mais próxima do centroide
        distances = euclidean_distances([cluster_mean], cluster_data)
        closest_sample_index_in_cluster = np.argmin(distances)
        
        # Converte para o índice no array original
        original_index = cluster_indices[closest_sample_index_in_cluster]
        selected_samples.append(original_index)
    
    return selected_samples

def partitions_sampling(data_file:str, section_type:str, train_valid_test_split:list):

    random.seed(42)
    
    data = np.load(data_file, mmap_mode='r')

    n_poolings = 5

    if section_type == 'inline':
        n_samples = data.shape[0]
    if section_type == 'crossline':
        n_samples = data.shape[1]
    if section_type == 'timeslice':
        n_samples = data.shape[2]

    n_train_samples = int(round(n_samples * train_valid_test_split[0]))
    n_val_samples = int(round(n_samples * train_valid_test_split[1]))
        
    # obs: não precisa transpor os inlines
    if section_type == 'crossline':
        data = np.transpose(data, (1, 0, 2))
    if section_type == 'timeslice':
        data = np.transpose(data, (2, 0, 1))
    
    data = data.reshape(data.shape[0], -1)
    samples_train = spectral_clustering_sampling(data, n_train_samples)
    samples_train.sort()

    samples_val = []
    for i in range(len(samples_train) - 1):
        middle_sample = (samples_train[i] + samples_train[i+1]) // 2
        samples_val.append(middle_sample)
    
    samples_train_val = samples_train + samples_val
    samples_train_val.sort()
    
    samples_test = [i for i in range(n_samples) if i not in samples_train_val]

    return sorted(samples_train), sorted(samples_val), sorted(samples_test)