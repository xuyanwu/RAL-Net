import torch
from torch.autograd import Variable
import numpy as np
import time
eps = 1e-6

def Hardloss(eye,f1,f2):
    l = int(f1.size(0))
    eyes = eye[0:l,0:l]

    MM=torch.mm(f1,torch.t(f2))
    MM= (1-MM)*2
    D = MM * 1
    D1 = D * 1 + eyes
    # D1 = D1.view(-1)
    # _,index = torch.topk(D1,12,dim=0,largest=False)
    # D1[index]=4
    # D1 = D1.view(l,-1)
    mask = (D1.ge(0.0000064).float()-1.0)*(-1)
    mask = mask.type_as(D1)*10
    D1 = D1+mask

    D_diag = torch.diag(D)
    D_diag = torch.unsqueeze(D_diag,dim=1)


    rc_min = torch.cat((D1,torch.t(D1)),dim = 1)

    triplet_min,_ = torch.min(rc_min,1)

    triplet_min = torch.unsqueeze(triplet_min,dim=1)

    L_max = D_diag - torch.clamp(triplet_min - 1.5,max=0)
    # L = torch.clamp(L_max, min=0)

    Hardloss = torch.sum(L_max) / l

    return Hardloss




