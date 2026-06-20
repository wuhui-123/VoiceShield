import torch
import torch.nn as nn
import numpy as np

from torch.nn.functional import binary_cross_entropy_with_logits as BCELogits
from sklearn.metrics import roc_curve

def compute_logits(output_spikes):
    logits = output_spikes.sum(dim=0).mean(dim=(1, 2))
    return logits

def compute_eer(labels, scores):
    if np.max(scores) > 1 or np.min(scores) < 0:
        scores = 1 / (1 + np.exp(-scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold

def compute_min_dcf(labels, scores, p_target=0.01, c_miss=1, c_fa=1):
    if np.max(scores) > 1 or np.min(scores) < 0:
        scores = 1 / (1 + np.exp(-scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf = np.min(dcf)
    min_dcf_normalized = min_dcf / min(c_miss * p_target, c_fa * (1 - p_target))
    return min_dcf_normalized

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = BCELogits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * ce_loss
        return loss.mean()