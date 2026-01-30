from nets.UNet import Unet
from nets.ResUNet import ResUnet
from nets.DeepLab import DeepLabV3Plus
from nets.DNFSLima2024 import DNFS
from nets.UNetLiu2024 import Unet77
from nets.Vit import Vit
from nets.Segformer import Segformer
from utils.BoundariesDataset import MaskedLabelSubset, load_subset, normalization
from utils.BoundariesTrain import train_validate
from utils.Losses import DistanceTransformLoss, combined_boundary_and_dice, combined_bce_and_dice, combined_bce_and_jaccard, FocalLoss, DNFS_combined_bce_and_jaccard
from selection_methods import Ampook, EquallySpaced, KFold, Random, SpectralClustering, Sequential
from selection_methods.select_samples import sampling

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from random import shuffle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:32'

data_file = '/pgeoprj/godeep/ewac_2/seismic_patterns/datasets/public_real_data/penobscot_dataset_cropped.npy'
boundaries_label_file = '/pgeoprj/godeep/ewac_2/seismic_patterns/datasets/public_real_data/penobscot_boundaries_cropped.npy'
boundaries_pred_file = '/pgeoprj/godeep/ewac_2/seismic_patterns/datasets/models_predictions/penobscot_15c_boundaries_prediction.npy'
output_dir = '/pgeoprj/godeep/ewac_2/seismic_patterns/datasets/models'
output_name = 'penobscot_16e'
data_preprocessing = normalization
sampling_method = EquallySpaced # [Ampook, EquallySpaced, KFold, Random, SpectralClustering, Sequential]
train_valid_test_split = [0.05, 0.05, 0.9]
n_folds = None ### No caso de optar por KFold
current_fold = None ### No caso de optar por KFold
section_type = 'inline'
loss_function = nn.BCEWithLogitsLoss() # [nn.BCEWithLogitsLoss(), DistanceTransformLoss, combined_boundary_and_dice, combined_bce_and_dice, combined_bce_and_jaccard, FocalLoss(), DNFS_combined_bce_and_jaccard]
batch_size = 8
epochs = 500
patience = 500 # IMPORTANTE! Após patience epochs, se o modelo não evoluir o treinamento é interrompido (obs: o modelo salvo não é do do momento do early stopping, e sim o melhor)
learning_rate = 0.001
kernel_size = 3
model = Unet
train_samples, validation_samples, _ = sampling(data_file, sampling_method, section_type, train_valid_test_split, n_folds, current_fold)

def main_train_validate(rank,
                        is_distributed,
                        world_size, model,
                        kernel_size,
                        train_data,
                        train_label,
                        train_pred,
                        validation_data,
                        validation_label,
                        validation_pred,
                        batch_size,
                        epochs,
                        patience,
                        learning_rate,
                        loss_function,
                        output_dir,
                        output_name):
    
    experiment_dir = f'{output_dir}/{output_name}'
    os.makedirs(experiment_dir, exist_ok=True)
        
    if is_distributed:
        rank = rank
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        if rank != 'cpu':
            rank = f'cuda:{rank}'

    train_dataset = MaskedLabelSubset(train_data, train_label, rank, data_augmentation=False, mask_pred=True, subset_pred=train_pred)
    val_dataset = MaskedLabelSubset(validation_data, validation_label, rank, data_augmentation=False, mask_pred=False, subset_pred=validation_pred)
    
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    if rank == 0 or is_distributed == False:
        print('Train dataset samples: ', len(train_dataset))
        print('Number of train batches: ', len(train_dataloader))
        print('Validation dataset samples: ', len(val_dataset))
        print('Number of validation batches: ', len(val_dataloader))

    model = model(kernel_size, input_channels=2).to(rank)
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_validate(is_distributed, model, optimizer, loss_function, epochs, patience, rank, experiment_dir, train_dataloader, val_dataloader)

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"N GPUs: {world_size}", flush=True)
        if world_size == 1:
            rank = 0
            is_distributed = False
        else:
            is_distributed = True
    else:
        world_size = 1
        is_distributed = False
        rank = 'cpu'
        print("Device: CPU.\nWarning: This process running in this device is quite slow.", flush=True)

        print(f'''
data_file: {data_file}
boundaries_label_file: {boundaries_label_file}
boundaries_pred_file: {boundaries_pred_file}
output_dir: {output_dir}
output_name: {output_name}
data_preprocessing: {data_preprocessing}
sampling_method: {sampling_method}
train_valid_test_split: {train_valid_test_split}
n_folds: {n_folds}
current_fold: {current_fold}
section_type: {section_type}
loss_function: {loss_function}
batch_size: {batch_size}
epochs: {epochs}
patience: {patience}
learning_rate: {learning_rate}
kernel_size: {kernel_size}
model: {model.__name__}
''', flush=True)

    train_data, train_label, train_pred, _ = load_subset(data_file,
                                                         boundaries_label_file,
                                                         section_type,
                                                         partition='train',
                                                         samples_partition=train_samples,
                                                         pred_file=boundaries_pred_file)

    train_data = data_preprocessing(train_data)
    
    validation_data, validation_label, validation_pred, _ = load_subset(data_file,
                                                                        boundaries_label_file,
                                                                        section_type,
                                                                        partition='validation',
                                                                        samples_partition=validation_samples,
                                                                        pred_file=boundaries_pred_file)
    validation_data = data_preprocessing(validation_data)

    args = (rank,
            is_distributed,
            world_size,
            model,
            kernel_size,
            train_data,
            train_label,
            train_pred,
            validation_data,
            validation_label,
            validation_pred,
            batch_size,
            epochs,
            patience,
            learning_rate,
            loss_function,
            output_dir,
            output_name)

    if is_distributed:
        mp.spawn(main_train_validate, args=args[1:], nprocs=world_size, join=True)
    else:
        main_train_validate(*args)
