import numpy as np
import torch
from skimage.morphology import disk, dilation
from scipy.ndimage import distance_transform_edt

def compute_dist_transform_error(targets_borders, predictions_borders):

    if isinstance(targets_borders, np.ndarray) or isinstance(predictions_borders, np.ndarray):
        targets_borders = torch.from_numpy(targets_borders)
        predictions_borders = torch.from_numpy(predictions_borders)
    
    def get_distance_matrix(matrix):
        _, _, height, width = matrix.shape
        
        # Initialize distance matrix with max penalty
        penalty = torch.tensor(height).sum()
        dist_matrix = torch.ones_like(matrix).float() * penalty

        # Replace border values with zero
        pos_indices = torch.nonzero(matrix == 1, as_tuple=True)
        dist_matrix[pos_indices] = 0
        
        # Iterate over matrix rows to create distance mapping
        for index in range(1, height): 
            dist_matrix[:,:,index,:] = torch.min(dist_matrix[:,:,index,:], dist_matrix[:,:,index-1,:]+1)
        for index in reversed(range(0, height-1)): 
            dist_matrix[:,:,index,:] = torch.min(dist_matrix[:,:,index,:], dist_matrix[:,:,index+1,:]+1)
            
        return dist_matrix

    distance_matrix = get_distance_matrix(targets_borders)
    
    prediction_error = (distance_matrix * predictions_borders).sum() / distance_matrix.numel()
    return prediction_error.item()
    
def binary_metrics(ground_truth_batch, output_batch, weighted_metrics=False):

    B, C, W, H = ground_truth_batch.shape

    # cálculo feito antes da conversão para numpy array, pois o cálculo da dist_transfor é feito em torch
    dist_pred_label = compute_dist_transform_error(ground_truth_batch, output_batch)
    dist_label_pred = compute_dist_transform_error(output_batch, ground_truth_batch)
    bilateral_dist_transform_error = (dist_pred_label + dist_label_pred) / 2
    
    if isinstance(ground_truth_batch, torch.Tensor) or isinstance(output_batch, torch.Tensor):
        ground_truth_batch = ground_truth_batch.detach().cpu().numpy().squeeze(1)
        output_batch = output_batch.detach().cpu().numpy().squeeze(1)
    else:
        ground_truth_batch = ground_truth_batch.squeeze(1)
        output_batch = output_batch.squeeze(1)

    if weighted_metrics:
        dilated_ground_truth_batch = ground_truth_batch.copy()
        for i in range(dilated_ground_truth_batch.shape[0]):
            distance = distance_transform_edt(dilated_ground_truth_batch[i] == 0)
            # Define a função de decaimento: por exemplo, exponencial ou linear
            # Aqui usamos decaimento exponencial com base 0.5 por pixel
            distance = np.where(dilated_ground_truth_batch[i] == 1, 1.0, 0.5 ** distance)
            # IMPORTANTE! Corta a expansão em uma distância máxima
            max_distance = 4
            distance[distance > max_distance] = 0
            dilated_ground_truth_batch[i] = distance
    
        dilated_ground_truth_batch = dilated_ground_truth_batch.ravel()

    ground_truth_batch = ground_truth_batch.ravel()
    output_batch = output_batch.ravel()
    
    if weighted_metrics:
        TP = np.sum(dilated_ground_truth_batch * output_batch)
        FP = np.sum((dilated_ground_truth_batch == 0) & (output_batch == 1))
    else:
        TP = np.sum((ground_truth_batch == 1) & (output_batch == 1))
        FP = np.sum((ground_truth_batch == 0) & (output_batch == 1))

    # IMPORTANTE: Para cálculo dos negativos (seja falsos ou verdadeiros) considera-se a label original
    # Caso contrário qualquer pixel na região dos limites expandidos onde a predição é zero seria considerado como um erro de predição
    # E isso não é verdade, pois ali de fato seriam pixels negativos na label verdadeira
    TN = np.sum((ground_truth_batch == 0) & (output_batch == 0))
    FN = np.sum((ground_truth_batch == 1) & (output_batch == 0))

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    fallout = FP / (FP + TN) if (FP + TN) != 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return bilateral_dist_transform_error, accuracy, precision, recall, fallout, iou, f1_score