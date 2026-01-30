import numpy as np
import time
from sklearn.cluster import KMeans
import random
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')

def apply_padding(sections):

    def next_multiple_of_32(n):
        if n % 32 == 0:
            return n
        return (n // 32 + 1) * 32

    C, H, W = sections.shape

    H_pad = next_multiple_of_32(H)
    W_pad = next_multiple_of_32(W)

    pad_h = H_pad - H
    pad_w = W_pad - W

    if pad_h == 0 and pad_w == 0:
        return sections

    # padding por seção (aplicado igual para cada canal)
    padded = np.pad(
        sections,
        pad_width=((0, 0), (0, pad_h), (0, pad_w)),
        mode='constant'
    )

    return padded

def max_pool(sections):

    C, H, W = sections.shape
    block_height, block_width = 2, 2

    sec = sections.reshape(
        C,
        H // block_height, block_height,
        W // block_width, block_width
    )

    # (C, H/2, W/2, 4)
    sec_flat = sec.transpose(0, 1, 3, 2, 4).reshape(
        C, H // block_height, W // block_width, 4
    )

    # índice do maior em módulo
    idx = np.abs(sec_flat).argmax(axis=-1)

    # seleção
    out = np.take_along_axis(sec_flat, idx[..., None], axis=-1)[..., 0]

    return out

def kmeans_clustering(all_samples, n_clusters):
    start = time.time()
    print(f'Clustering data with k-means')
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_samples)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    closest_sample_indices = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_samples = all_samples[cluster_indices]

        distances = cdist(cluster_samples, [centroids[i]], metric='euclidean')

        closest_idx = np.argmin(distances)

        closest_sample_indices.append(cluster_indices[closest_idx])
    
    return closest_sample_indices

def partitions_sampling(data_file:str, section_type:str, train_valid_test_split:list, chunk_size:int):

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
        
    if section_type == 'inline':
        
        # Passa as duas primeiras seçoes pelo processo para saber o tamanho necessário do array que armazenará as seções reduzidas
        example_data = data[0:2, :, :]
        example_data = apply_padding(example_data)
        for _ in range(n_poolings):
            example_data = max_pool(example_data)

        all_pooled_data = np.zeros((data.shape[0], example_data.shape[1], example_data.shape[2]))
        
        n_sections = data.shape[0]
        chunks_limits = list(range(0, n_sections, chunk_size))
        # para incluir o ultimo chunk, em caso de o numero de seçoes nao ser divisivel pelo numero de chunks definido
        if chunks_limits[-1] != n_sections:
            chunks_limits.append(n_sections)

        print("Reducing images for clustering, please wait...")
        for idx in range(len(chunks_limits) - 1):
            print(f'Chunk {idx + 1}/{len(chunks_limits)}')

            sections = range(chunks_limits[idx], chunks_limits[idx + 1])
            pooled = data[sections, :, :]
            pooled = apply_padding(pooled)

            for _ in range(n_poolings): # 5 é o número de reduções/poolings que o dado passará
                pooled = max_pool(pooled)

            all_pooled_data[sections, :, :] = pooled
            
        print("All images pooled")

    if section_type == 'crossline':
        
        # Passa as duas primeiras seçoes pelo processo para saber o tamanho necessário do array que armazenará as seções reduzidas
        example_data = data[:, 0:2, :]
        example_data = np.transpose(example_data, (1, 0, 2)) # e necessario transpor no caso de crosslines e timeslices
        example_data = apply_padding(example_data)

        for _ in range(n_poolings):
            example_data = max_pool(example_data)

        all_pooled_data = np.zeros((data.shape[1], example_data.shape[1], example_data.shape[2]))
        
        n_sections = data.shape[1]
        chunks_limits = list(range(0, n_sections, chunk_size))
        # para incluir o ultimo chunk, em caso de o numero de seçoes nao ser divisivel pelo numero de chunks definido
        if chunks_limits[-1] != n_sections:
            chunks_limits.append(n_sections)

        print("Reducing images for clustering, please wait...")
        for idx in range(len(chunks_limits) - 1):
            print(f'Chunk {idx + 1}/{len(chunks_limits)}')

            sections = range(chunks_limits[idx], chunks_limits[idx + 1])
            pooled = data[:, sections, :]
            pooled = np.transpose(pooled, (1, 0, 2)) # e necessario transpor no caso de crosslines e timeslices
            pooled = apply_padding(pooled)

            for _ in range(n_poolings): # 5 é o número de reduções/poolings que o dado passará
                pooled = max_pool(pooled)

            all_pooled_data[sections, :, :] = pooled
            
        print("All images pooled")

    if section_type == 'timeslice':
        
        # Passa as duas primeiras seçoes pelo processo para saber o tamanho necessário do array que armazenará as seções reduzidas
        example_data = data[:, :, 0:2]
        example_data = np.transpose(example_data, (2, 0, 1)) # e necessario transpor no caso de crosslines e timeslices
        example_data = apply_padding(example_data)

        for _ in range(n_poolings):
            example_data = max_pool(example_data)

        all_pooled_data = np.zeros((data.shape[2], example_data.shape[1], example_data.shape[2]))
        
        n_sections = data.shape[2]
        chunks_limits = list(range(0, n_sections, chunk_size))
        # para incluir o ultimo chunk, em caso de o numero de seçoes nao ser divisivel pelo numero de chunks definido
        if chunks_limits[-1] != n_sections:
            chunks_limits.append(n_sections)

        print("Reducing images for clustering, please wait...")
        for idx in range(len(chunks_limits) - 1):
            print(f'Chunk {idx + 1}/{len(chunks_limits)}')

            sections = range(chunks_limits[idx], chunks_limits[idx + 1])
            pooled = data[:, :, sections]
            pooled = np.transpose(pooled, (2, 0, 1)) # e necessario transpor no caso de crosslines e timeslices
            pooled = apply_padding(pooled)

            for _ in range(n_poolings): # 5 é o número de reduções/poolings que o dado passará
                pooled = max_pool(pooled)

            all_pooled_data[sections, :, :] = pooled
            
        print("All images pooled")
        
    all_pooled_data = all_pooled_data.reshape(all_pooled_data.shape[0], -1)
    samples_train = kmeans_clustering(all_pooled_data, n_clusters=n_train_samples)
    samples_train.sort()

    samples_val = []
    for i in range(len(samples_train) - 1):
        middle_sample = (samples_train[i] + samples_train[i+1]) // 2
        samples_val.append(middle_sample)
    
    samples_train_val = samples_train + samples_val
    samples_train_val.sort()
    
    samples_test = [i for i in range(n_samples) if i not in samples_train_val]

    return sorted(samples_train), sorted(samples_val), sorted(samples_test)
    