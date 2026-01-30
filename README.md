# STAR: Facies boundaries delineation with few labeled samples

This repository contains the implementations of the pipeline proposed in the study:

"STAR: A robust deep learning pipeline for facies boundaries delineation using few labeled samples."

<!------------------------------------------------------------------------------------------------->

# Overview

STAR is an experimental framework for precise seismic facies boundary delineation. The best components obtained through STAR integrates (1) AMPooK selection strategy, (2) DeepLabV3+ as segmentation model, trained with Distance Transform Loss (DTL), and (3) a refinement stage based on masked and denoising autoencoders. Performance is evaluated using metrics sensitive to small spatial misalignments: BDTE, distance-weighted IoU (DWIoU) and F₁-score (DWF₁).

All components evaluated within STAR are fully implemented and available for use through the provided code.

Sampling techniques:
- Sequential
- Equally spaced
- Random
- Spectral clustering (SCS)
- Absolute-Max-Pooling combined with K-means clustering (AMPooK)
  
Loss functions:
- BCE loss
- Focal loss
- Distance transform loss (DTL)
- BCE + jaccard loss
- Boundary + dice
- Boundary + dice loss

Network architectures:
- U-Net
- ResUNet
- DeepLabV3+
- DNFS
- U-Net 7×7

Autoencoders architectures for refinement:
- U-Net
- ResUNet
- DeepLabV3+
- DNFS
- U-Net 7×7

<!------------------------------------------------------------------------------------------------->

# How to use

## Training and testing models

The [codes](codes) directory contains 4 scripts required to run training, testing, and refinement experiments.
Each script is self-contained and allows independent configuration of parameters and hyperparameters.

- train_supervised.py
Script for training a model from scratch.
Inputs: a seismic volume and a binary facies-boundary label volume, both provided as NumPy arrays.
Outputs: the trained model saved in .pth format, along with two .json files containing per-epoch training and validation metrics.
Additionally, log files report information about hyperparameters, training time, and execution status.

- train_refinement.py
Script for refining a pre-trained model.
Inputs: a pre-trained model in .pth format, a seismic volume, and a volume containing the binary predictions of the pre-trained model, all provided as NumPy arrays.
Outputs: the refined model saved in .pth format, along with two .json files containing per-epoch training and validation metrics.
Log files also provide details on hyperparameters, training time, and execution status.

- test_supervised.py
Script for evaluating a model trained from scratch.
Inputs: a seismic volume and a binary facies-boundary label volume, both provided as NumPy arrays.
Outputs: a volume containing the model predictions, along with log files reporting the evaluation metrics and information about the hyperparameters.

- test_refinement.py
Script for evaluating a model obtained through the refinement stage.
Inputs: a seismic volume and a volume containing the binary predictions of the pre-trained model, both provided as NumPy arrays.
Outputs: a volume containing the refined predictions, along with log files reporting the evaluation metrics and information about the hyperparameters.

<!------------------------------------------------------------------------------------------------->

## Preprocessing and visualization notebooks

The [preprocessing_and_visualization](codes/preprocessing_and_visualization) directory contains notebooks used for data preprocessing and preparation prior to running the experiments, as well as modules for visualizing training progress, data, labels, and model predictions.

- plot_metrics_per_epoch.ipynb
Enables real-time monitoring of the training process by reading the .json files generated during training.
Displays the evolution of the loss function and evaluation metrics across epochs.

- boundaries_extraction.ipynb
Extracts binary facies boundaries from a labeled facies volume.

- spurious_data_removing.ipynb
Used to remove seismic sections containing spurious data or inconsistent labels.
Note: the sections to be removed must be manually specified after prior analysis.

- seismic_visualizer.ipynb
Provides visualization of NumPy-based volumes, including seismic data, labels, and model predictions.

- tridimensional_view_interface.ipynb
Allows interactive visualization of a 3D volume using three orthogonal planes and a 3D surface representation.


