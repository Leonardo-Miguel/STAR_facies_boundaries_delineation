from nets.UNet import Unet
from nets.ResUNet import ResUnet
from nets.DeepLab import DeepLabV3Plus
from nets.DNFSLima2024 import DNFS
from nets.UNetLiu2024 import Unet77
from nets.Vit import Vit
from nets.Segformer import Segformer
from utils.BoundariesDataset import SeismicSubset, load_subset, normalization
from utils.BoundariesPrediction import predict_boundaries
from selection_methods import Ampook, EquallySpaced, KFold, Random, SpectralClustering, Sequential
from selection_methods.select_samples import sampling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:32'

##################### Input parameters #######################

# Input files
data_file = 'datasets/data.npy'
boundaries_label_file = 'datasets/boundaries_labels.npy'

# Trained model
model_file = 'models/experiment_01/complete_model.pth'

# Output configuration
output_dir = 'predictions'
output_name = 'experiment_01'

# Sampling strategy
sampling_method = Ampook  
# Options: Ampook, EquallySpaced, KFold, Random, SpectralClustering, Sequential

# Dataset split ratios (train / validation / test)
train_valid_test_split = [0.05, 0.05, 0.90]

# K-Fold parameters (used only if sampling_method == KFold)
n_folds = None
current_fold = None

# Data configuration
section_type = 'inline'  # or 'crossline'
batch_size = 16

# Training hyperparameters
learning_rate = 1e-3
kernel_size = 3
model = UNet

# Output options
save_predictions = True

############################################################################################

if __name__ == "__main__":

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"N GPUs: {world_size}")
        if world_size == 1:
            rank = 'cuda:0'
        else:
            print("This is a single GPU (non-distributed) or CPU process. Please check this parameter. Ending program.")
            sys.exit()
    else:
        rank = 'cpu'
        print("Device: CPU.")

    volume_shape = np.load(data_file, mmap_mode='r').shape
    _, _, test_samples = sampling(data_file, sampling_method, section_type, train_valid_test_split, n_folds, current_fold)
    
    print(f'''
data_file: {data_file}
boundaries_label_file: {boundaries_label_file}
model_file: {model_file}
output_dir: {output_dir}
output_name: {output_name}
save_predictions: {save_predictions}
volume_shape: {volume_shape}
sampling_method: {sampling_method}
train_valid_test_split: {train_valid_test_split}
n_folds: {n_folds}
current_fold: {current_fold}
section_type: {section_type}
batch_size: {batch_size}
learning_rate: {learning_rate}
kernel_size: {kernel_size}
model: {model.__name__}\n
test_samples: {test_samples}\n''')

    if save_predictions:
        os.makedirs(output_dir, exist_ok=True)

    test_data, test_label, _ = load_subset(data_file,
                                             boundaries_label_file,
                                             section_type,
                                             partition='test',
                                             samples_partition=test_samples)
    test_data = normalization(test_data)
    
    test_dataset = SeismicSubset(test_data, test_label, rank, data_augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    if model == Vit or model == Segformer:
        model = model().to(rank)
    else:
        model = model(kernel_size).to(rank)

    checkpoint = torch.load(model_file, map_location=rank)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    predict_boundaries(model, test_dataloader, test_samples, section_type, volume_shape, output_dir, output_name, save_predictions, rank)
