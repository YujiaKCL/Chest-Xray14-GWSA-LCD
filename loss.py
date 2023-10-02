import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import BinaryCrossEntropy


class MulBCE(nn.Module):
    def __init__(self, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.BCE = BinaryCrossEntropy(smoothing=label_smoothing, reduction=reduction)

    def forward(self, logits, labels):
        if len(labels.shape) > 1:
            labels = labels.flatten()
            print(labels.size())
        if len(logits.shape) > 1:
            logits = logits.flatten()
            print(logits.size())

        loss = self.BCE(logits, labels)
        return loss

class ClassWiseBCE(nn.Module):
    def __init__(self, num_tasks=5, pos_weights=None):
        if pos_weights is None:
            self.pos_weights = [1,] * num_tasks
        self.pos_weights = pos_weights
    
    def forward(self, logits, labels, task_id):
        return F.binary_cross_entropy_with_logits(
                logits[:, task_id], labels[:, task_id], pos_weight=self.pos_weights[task_id])
        
        