import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")


def partitions_sampling(data_file:str, section_type:str, train_valid_test_split:list):

    data = np.load(data_file, mmap_mode='r')

    if section_type == 'inline':
        n_samples = data.shape[0]
    if section_type == 'crossline':
        n_samples = data.shape[1]
    if section_type == 'timeslice':
        n_samples = data.shape[2]

    n_train_samples = int(round(n_samples * train_valid_test_split[0]))
    n_val_samples = int(round(n_samples * train_valid_test_split[1]))

    random.seed(42) # para as partições serem sempre iguais, ainda que aleatórias
    samples_positions = list(range(n_samples))
    random.shuffle(samples_positions)

    samples_train = samples_positions[:n_train_samples]
    samples_val = samples_positions[n_train_samples:n_train_samples + n_val_samples]
    samples_test = samples_positions[n_train_samples + n_val_samples:]

    return sorted(samples_train), sorted(samples_val), sorted(samples_test)
