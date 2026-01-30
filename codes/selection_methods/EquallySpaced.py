import numpy as np

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

    if n_train_samples > 0:
        step = int(round(n_samples / n_train_samples))
        samples_train = list(range(0, n_samples, step))
        #slice necessário porque dependendo do passo e do número de amostras totais, a partição terá mais amostras que o esepcificado
        samples_train = samples_train[:n_train_samples]
        first_sample_val = samples_train[1] - (step // 2)

        samples_val = range(first_sample_val, n_samples, step)
        samples_val = [i for i in samples_val if i not in samples_train]
        samples_val = samples_val[:n_val_samples] #slice pelo mesmo motivo do slice do treino
    else: 
        samples_train = []
        samples_val = []
    
    samples_train_val = samples_train + samples_val
    samples_train_val.sort()
    
    samples_test = [i for i in range(n_samples) if i not in samples_train_val]

    return sorted(samples_train), sorted(samples_val), sorted(samples_test)
