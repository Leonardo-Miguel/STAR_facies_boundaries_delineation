import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import random
from skimage import exposure

import warnings
warnings.filterwarnings("ignore")
    
def normalization(data:np.array) -> np.array:

    print(f'Original data range: [{np.min(data)} , {np.max(data)}]', flush=True)
    
    mean = np.mean(data)
    std_dev = np.std(data)
    data = (data - mean) / std_dev # standardization
    
    data = np.clip(data, -3, 3) # Clip the outliers

    min_val = np.min(data)
    max_val = np.max(data)

    data = 2 * (data - min_val) / (max_val - min_val) - 1

    print(f'Normalized data range: [{np.min(data)}, {np.max(data)}]\n', flush=True)
    
    return data

def load_subset(data_file:str, label_file:str, section_type:str, partition:str, samples_partition:list, pred_file:str=None):
    
    data = np.load(data_file, mmap_mode='r')
    if label_file is not None:
        label = np.load(label_file, mmap_mode='r')
    if pred_file is not None:
        pred = np.load(pred_file, mmap_mode='r')

    if partition != 'test':
        if section_type == 'inline':
            subset_data = data[samples_partition, :, :]
            if label_file is not None:
                subset_label = label[samples_partition, :, :]
            if pred_file is not None:
                subset_pred = pred[samples_partition, :, :]
                
        if section_type == 'crossline':
            subset_data = data[:, samples_partition, :]
            if label_file is not None:
                subset_label = label[:, samples_partition, :]
            if pred_file is not None:
                subset_pred = pred[:, samples_partition, :]
                
        if section_type == 'timeslice':
            subset_data = data[:, :, samples_partition]
            if label_file is not None:
                subset_label = label[:, :, samples_partition]
            if pred_file is not None:
                subset_pred = pred[:, :, samples_partition]
                
    if partition == 'test':
        subset_data = data[:, :, :]
        if label_file is not None:
            subset_label = label[:, :, :]
        if pred_file is not None:
            subset_pred = pred[:, :, :]

    if label_file is not None and pred_file is not None:
        return subset_data, subset_label, subset_pred, samples_partition
    elif label_file is not None and pred_file is None:
        return subset_data, subset_label, samples_partition
    else:
        return subset_data, samples_partition

class SaltAndPepper(A.ImageOnlyTransform):
    def __init__(self, prob=0.01, always_apply=False, p=0.5):
        super(SaltAndPepper, self).__init__(always_apply, p)
        self.prob = prob

    def apply(self, img, **params):
        noisy = img.copy()
        num_salt = np.ceil(self.prob * img.size * 0.5)
        num_pepper = np.ceil(self.prob * img.size * 0.5)

        # salt (branco)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = np.max(noisy)

        # pepper (preto)
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = np.min(noisy)

        return noisy

class SeismicSubset(Dataset):
    
    def __init__(self, subset_data:np.array, subset_label:np.array, rank:str, data_augmentation=False):
        self.subset_data = subset_data
        self.subset_label = subset_label
        self.rank = rank
        self.data_augmentation = data_augmentation

    def apply_augmentation(self, sample):
        transform = A.Compose([
            A.ColorJitter(p=0.4),
            A.GaussianBlur(p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            SaltAndPepper(p=0.5),
            A.CoarseDropout(max_holes=60, max_height=0.03, max_width=0.03, fill_value=0, p=0.4)
        ])
        return transform(image=sample)['image']

    def __len__(self):
        return self.subset_data.shape[0]

    def __getitem__(self, idx):

        sample = self.subset_data[idx, :, :]
        if self.data_augmentation:
            sample = self.apply_augmentation(sample)
        sample = torch.from_numpy(sample).to(self.rank)
        sample = sample.unsqueeze(0)

        if self.subset_label is not None:
            label = self.subset_label[idx, :, :]
            label = torch.from_numpy(label).to(self.rank)
            label = label.unsqueeze(0)
            return sample, label
        else:
            return sample

class MaskedLabelSubset(Dataset):
    
    def __init__(self, subset_data:np.array, subset_label:np.array, rank:str, data_augmentation=False, mask_pred=False, subset_pred:np.array=None):
        self.subset_data = subset_data
        self.subset_label = subset_label
        self.subset_pred = subset_pred
        self.rank = rank
        self.data_augmentation = data_augmentation
        self.mask_pred = mask_pred

    def apply_augmentation(self, sample):
        transform = A.Compose([
            A.ColorJitter(p=0.4),
            A.GaussianBlur(p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            SaltAndPepper(p=0.5),
            #A.CoarseDropout(max_holes=60, max_height=0.03, max_width=0.03, fill_value=0, p=0.4)
        ])
        return transform(image=sample)['image']
        
    def apply_mask(self, sample):
        transform = A.Compose([
            A.CoarseDropout(max_holes=10, max_height=0.10, max_width=0.10, fill_value=0, p=1),
            A.CoarseDropout(max_holes=500, max_height=0.02, max_width=0.02, fill_value=0, p=1),
            A.CoarseDropout(max_holes=25, max_height=0.002, max_width=0.02, fill_value=1, p=1)
        ])
        return transform(image=sample)['image']

    def __len__(self):
        return self.subset_data.shape[0]

    def __getitem__(self, idx):

        sample = self.subset_data[idx, :, :]
        if self.data_augmentation:
            sample = self.apply_augmentation(sample)
        sample = torch.from_numpy(sample).to(self.rank)
        sample = sample.unsqueeze(0)

        original_label = self.subset_label[idx, :, :]        
        original_label = torch.from_numpy(original_label).to(self.rank)
        original_label = original_label.unsqueeze(0)

        pred = self.subset_pred[idx, :, :]
        if self.mask_pred:
            pred = self.apply_mask(pred)
        pred = torch.from_numpy(pred).to(self.rank)
        pred = pred.unsqueeze(0)

        return torch.cat([sample, pred], dim=0), original_label