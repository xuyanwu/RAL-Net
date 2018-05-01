'''this Hard_loss is intercepted from HardNet
"Working hard to know your neighbor's margins: Local descriptor learning loss" '''

import torch
import torch.nn as nn
import sys

def distance_matrix_vector(f1, f2):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    MM = torch.mm(f1, torch.t(f2))
    MM = (1 - MM) * 2


    eps = 1e-6
    return torch.sqrt(MM+eps)

def loss_HardNet(eye,anchor, positive):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    l = int(dist_matrix.size(0))
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye[0:l,0:l]*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    min_neg = torch.min(dist_without_min_on_diag,1)[0]
    min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
    min_neg = torch.min(min_neg,min_neg2)
    min_neg = min_neg
    pos = pos1
    loss = torch.clamp(1 + pos - min_neg, min=0.0)

    loss = torch.mean(loss)
    return loss
