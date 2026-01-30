import numpy as np

import warnings
warnings.filterwarnings("ignore")

def partitions_sampling(data_file:str, section_type:str, current_fold:int, n_folds:int):

    data = np.load(data_file, mmap_mode='r')

    if section_type == 'inline':
        n_samples = data.shape[0]
    if section_type == 'crossline':
        n_samples = data.shape[1]
    if section_type == 'timeslice':
        n_samples = data.shape[2]

    n_fold_samples = int(round(n_samples * (1 / n_folds)))

    samples_positions = list(range(n_samples))
    if current_fold < n_folds:
        samples_train = samples_positions[(current_fold-1) * n_fold_samples : current_fold * n_fold_samples]
        samples_val = samples_positions[current_fold * n_fold_samples : (current_fold+1) * n_fold_samples]
    else:
        # OBS: para o último fold a lógica é diferente, já que o bloco de treino são as ultimas amostras, e o de validação são as primeiras
        samples_train = samples_positions[(current_fold-1) * n_fold_samples : current_fold * n_fold_samples]
        samples_val = samples_positions[ : n_fold_samples]

    samples_train_val = samples_train + samples_val
    samples_train_val.sort()
    
    samples_test = [i for i in range(n_samples) if i not in samples_train_val]

    return sorted(samples_train), sorted(samples_val), sorted(samples_test)
