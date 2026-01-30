import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from scipy.ndimage import distance_transform_edt as edt
import numpy as np

class DistanceTransformLoss(torch.nn.Module):
    """
    DistanceTransformLoss: A loss function that emphasizes boundary segmentation by
    combining a standard cross-entropy loss with a weighted loss based on the distance
    transform of boundary pixels.

    The distance transform loss focuses on correctly classifying pixels near boundaries
    by assigning higher penalties to misclassified pixels closer to ground truth boundaries.
    """    
    def __init__(self, border_params=[0, 0.5, 1.0], reduction='mean'):
        """
        Initializes the DistanceTransformLoss class.

        Args:
            border_params (list, optional): List of parameters for border loss configuration:
                - border_params[0]: Dilation factor for distance transform (default: 0)
                - border_params[1]: Power to raise the border loss (default: 0.5)
                - border_params[2]: Weighting factor for the border loss (default: 1.0)
            reduction (str, optional): Reduction mode for the loss (default: 'sum')
        """
        super(DistanceTransformLoss, self).__init__()
        self.border_dilate =   int(border_params[0])
        self.border_power  = float(border_params[1])
        self.border_weight = float(border_params[2])
        self.reduction = reduction

        self.target_criterion = nn.BCEWithLogitsLoss()

    def get_distance_matrix(self, matrix, dist_transform=True):
        """
        Calculates the distance transform of a boundary mask.

        Args:
            matrix (torch.Tensor): Input tensor of shape (N, C, H, W) containing boundary mask.
            dist_transform (bool, optional): If True, performs distance transform.
                                             If False, returns the original matrix.

        Returns:
            torch.Tensor: Distance transform of the input matrix.
        """
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

    def border_criterion(self, predictions, targets):
        """
        Calculates the border loss based on distance transform.

        Args:
            predictions (torch.Tensor): Model predictions of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation labels of shape (N, C, H, W).

        Returns:
            torch.Tensor: The calculated border loss value.
        """

        predictions = F.logsigmoid(predictions).exp()
        predictions = (predictions > 0.5).to(torch.uint8)
        distance_matrix = self.get_distance_matrix(targets, dist_transform=True)
        
        border_penalty = predictions * distance_matrix

        if border_penalty.sum() == 0:
            return torch.tensor(0)
        else:
            # Ruduce penalty though mean or sum of all batch samples
            if self.reduction == 'mean':
                return border_penalty[border_penalty != 0].mean()
            elif self.reduction ==  'sum':
                return border_penalty[border_penalty != 0].sum()
            else:
                return border_penalty[border_penalty != 0]
         
    def forward(self, predictions, targets):
        """
        Calculates the DistanceTransformLoss.

        Args:
            predictions (torch.Tensor): Model predictions of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth segmentation labels of shape (N, H, W).

        Returns:
            torch.Tensor: The calculated DistanceTransformLoss value.
        """
        target_loss = self.target_criterion(predictions, targets)
        targets = targets.int()
        border_loss = self.border_criterion(predictions, targets)

        return target_loss + (self.border_weight * border_loss**self.border_power)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        """
        alpha: peso para a classe positiva (default=0.25, como no paper original)
        gamma: fator de foco (default=2.0)
        reduction: "mean" ou "sum"
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        predictions: logits do modelo (antes da sigmoid), shape [B, 1, H, W]
        targets: rótulos binários (0 ou 1), shape [B, 1, H, W]
        """
        # Converter logits em probabilidades (sigmoid)
        probs = torch.sigmoid(predictions)
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calcular BCE para cada pixel
        bce_loss = F.binary_cross_entropy(probs, targets, reduction="none")

        # P_t = p se y=1, (1-p) se y=0
        pt = probs * targets + (1 - probs) * (1 - targets)

        # Focal loss: α * (1 - pt)^γ * BCE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()
            
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        predictions: logits do modelo (antes da sigmoid), shape [B, 1, H, W]
        targets: rótulos binários (0 ou 1), shape [B, 1, H, W]
        """
        # Transformar logits em probabilidades
        predictions = F.logsigmoid(predictions).exp()

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        # fórmula do Dice
        dice = (2 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice  # como é loss, usamos 1 - dice
        
def combined_bce_and_dice(predictions, targets):
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    loss_bce = bce(predictions, targets)
    loss_dice = dice(predictions, targets)
    return loss_bce + loss_dice
    
class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        predictions: logits do modelo (antes da sigmoid), shape [B, 1, H, W]
        targets: rótulos binários (0 ou 1), shape [B, 1, H, W]
        """
        # Transformar logits em probabilidades
        predictions = F.logsigmoid(predictions).exp()

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou  # como é loss, usamos 1 - IoU

def combined_bce_and_jaccard(predictions, targets):
    bce = nn.BCEWithLogitsLoss()
    jaccard = JaccardLoss()

    loss_bce = bce(predictions, targets)
    loss_jaccard = jaccard(predictions, targets)
    return loss_bce + loss_jaccard

def DNFS_combined_bce_and_jaccard(predictions, targets):

    loss_weight = 0.75
    
    bce = nn.BCEWithLogitsLoss()
    jaccard = JaccardLoss()

    loss_bce = bce(predictions, targets)
    loss_jaccard = jaccard(predictions, targets)
    iou = (loss_jaccard - 1) * (-1) # IMPORTANTE! Para recuperar o valor do iou, antes da inversão indice de jaccard, para poder ponderar corretamente
    return (loss_weight * loss_bce) - ((1 - loss_weight) * iou)
    
def compute_sdf(target):
    """
    Computa o Signed Distance Map (SDF) para um rótulo binário.
    target: tensor numpy [H, W] com valores {0,1}
    Retorna: SDF numpy [H, W]
    """
    posmask = target.astype(bool)

    if posmask.any():
        # distância para o fundo
        negmask = ~posmask
        dist_out = edt(negmask)
        # distância para o objeto
        dist_in = edt(posmask)
        sdf = dist_out - dist_in
    else:
        sdf = np.zeros_like(target, dtype=np.float32)

    return sdf.astype(np.float32)

class BoundaryLoss(nn.Module):
    def __init__(self):
        """
        Implementação da Boundary Loss
        Paper: 'Boundary loss for highly unbalanced segmentation' (Kervadec et al., 2019)
        """
        super(BoundaryLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        predictions: logits do modelo (antes da sigmoid), shape [B, 1, H, W]
        targets: rótulos binários (0 ou 1), shape [B, 1, H, W]
        """
        probs = torch.sigmoid(predictions)

        B, _, H, W = targets.shape
        loss = 0.0

        for b in range(B):
            # converter GT para numpy e calcular SDF
            target_np = targets[b, 0].cpu().numpy().astype(np.uint8)
            sdf = compute_sdf(target_np)

            # converter de volta para tensor
            sdf_tensor = torch.from_numpy(sdf).to(predictions.device)

            # Boundary loss: soma(p * sdf)
            # aqui, p = probabilidade prevista pelo modelo
            pc = probs[b, 0]
            loss += torch.sum(pc * sdf_tensor) / (H * W)

        return loss / B

def combined_boundary_and_dice(predictions, targets):
    boundary = BoundaryLoss()
    dice = DiceLoss()

    loss_boundary = boundary(predictions, targets)
    loss_dice = dice(predictions, targets)
    return loss_boundary + loss_dice