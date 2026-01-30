from selection_methods import Ampook, EquallySpaced, KFold, Random, SpectralClustering, Sequential
from random import shuffle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

def sampling(data_file, sampling_method, section_type, train_valid_test_split, n_folds=None, current_fold=None):
    
    if sampling_method == KFold:
        samples = sampling_method.partitions_sampling(data_file, section_type, current_fold, n_folds)
    elif sampling_method == Ampook:
        chunk_size = 100
        samples = sampling_method.partitions_sampling(data_file, section_type, train_valid_test_split, chunk_size)
    else:
        samples = sampling_method.partitions_sampling(data_file, section_type, train_valid_test_split)

    samples_train, samples_val, samples_test = samples

    return samples_train, samples_val, samples_test

