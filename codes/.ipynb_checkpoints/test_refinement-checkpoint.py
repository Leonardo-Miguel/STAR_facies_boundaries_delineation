from nets.UNet import Unet
from nets.ResUNet import ResUnet
from nets.DeepLab import DeepLabV3Plus
from nets.DNFSLima2024 import DNFS
from nets.UNetLiu2024 import Unet77
from utils.BoundariesDataset import SeismicSubset, MaskedLabelSubset, load_subset, normalization
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

# Input data files
data_file = 'datasets/data.npy'                # file containing the seismic data
boundaries_label_file = 'datasets/labels.npy'  # file containing the boundary labels

# Predictions already made by a model
boundaries_pred_file = 'predictions/boundaries_prediction.npy'

# Trained model file
model_file = 'models/trained_model.pth'

# Output directory and experiment name
output_dir = 'predictions'
output_name = 'experiment_01'

# Dataset split ratios (train / validation / test)
train_valid_test_split = [0.05, 0.05, 0.90]

# Sampling strategy
sampling_method = Ampook # Options: Ampook, EquallySpaced, KFold, Random, SpectralClustering, Sequential

# K-Fold parameters (used only if sampling_method == KFold)
n_folds = None
current_fold = None

# Data configuration
section_type = 'inline'  # Options: 'inline', 'crossline', 'timeslice'
batch_size = 16

# Hyperparameters (it should be the same as those used in training)
learning_rate = 1e-3
kernel_size = 3
model = DeepLabV3Plus # Options: Unet, ResUnet, DeepLabV3Plus, DNFS, Unet77

# Output options
save_predictions = True

########################################################################

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
boundaries_pred_file: {boundaries_pred_file}
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

    test_data, test_label, test_pred, _ = load_subset(data_file,
                                             boundaries_label_file,
                                             section_type,
                                             partition='test',
                                             samples_partition=test_samples,
                                            pred_file=boundaries_pred_file)
    test_data = normalization(test_data)
    
    test_dataset = MaskedLabelSubset(test_data, test_label, rank, data_augmentation=False, mask_pred=False, subset_pred=test_pred)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    model = model(kernel_size, input_channels=2).to(rank)
    checkpoint = torch.load(model_file, map_location=rank)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    predict_boundaries(model, test_dataloader, test_samples, section_type, volume_shape, output_dir, output_name, save_predictions, rank)

