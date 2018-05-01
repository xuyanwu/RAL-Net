import torch
from torch.autograd import Variable
import numpy as np
import time
eps = 1e-6

def Dig_loss_triplet(eye,f1,f2):

    l = int(f1.size(0))
    eyes = eye[0:l,0:l]

    MM = torch.mm(f1, torch.t(f2))
    MM = (1 - MM) * 2

    D = MM * 1
    D_diag = torch.diag(D)
    # D_diag = torch.unsqueeze(D_diag,dim=1)


    D1 = D * 1 + eyes
    # D1 = D1.view(-1)
    # _,index = torch.topk(D1,12,dim=0,largest=False)
    # D1[index]=4
    # D1 = D1.view(l,-1)
    mask = (D1.ge(0.0001).float()-1.0)*(-1)
    mask = mask.type_as(D1)*10
    D1 = D1+mask

    l = int(D1.size(0))
    D2 = D1.data.cpu()


    row_min, row_index = torch.min(D2, 1)
    col_min, col_index = torch.min(D2, 0)
    row_min = torch.unsqueeze(row_min, dim=1)
    col_min = torch.unsqueeze(col_min, dim=1)

    rc_min = torch.cat((row_min, col_min), dim=1)
    triplet_min, min_index = torch.min(rc_min, 1)




    triplet_min1 = triplet_min.cpu()

    divide = []
    index = [[],[]]
    pos_index=[]

    for i in range(l):
        # print(i)
        a = min_index[i]
        if (a == 0):
            row_f = row_index[i]
            col_f = i
            row_f1 = row_index[i]
            col_f1 = i

            I = torch.t(D2)
            flag = -1
        elif (a == 1):
            row_f = col_index[i]
            col_f = i
            row_f1 = col_index[i]
            col_f1 = i

            I = D2
            flag = 1

        c = -1
        c_min = triplet_min1[i]
        j = 0
        while (c_min != c):
            if flag == -1:
                index[0].append(col_f1)
                index[1].append(row_f1)
            elif flag == 1:
                index[0].append(row_f1)
                index[1].append(col_f1)
            flag = flag*(-1)
            j = j + 1
            c = c_min
            # print(row_f)
            # print(c)
            a =0

            row_f1 = row_f
            c_min, col_f = torch.min(I[row_f], dim=0)
            c_min = c_min[0]
            col_f = col_f[0]

            col_f1 = col_f

            row_f = col_f
            I = torch.t(I)
        for k in range(j):
            divide.append(j)
            pos_index.append(i)

    divide = torch.unsqueeze(torch.from_numpy(np.array(divide)).float(),dim=1)
    divide = Variable(divide).cuda()
    pos = torch.unsqueeze(D_diag[pos_index],dim=1)
    # print(divide)
    # print(index)
    triplet_minM = torch.unsqueeze(D1[index],dim=1)
    # print(triplet_minM)
    triplet_minM = torch.div(torch.clamp((2+pos-triplet_minM),min=0),divide)

    loss = torch.sum(triplet_minM)/l
    return loss



