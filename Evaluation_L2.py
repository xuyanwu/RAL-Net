import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
from Utils import L2Norm
from sklearn.metrics import auc,roc_curve

def EvalL2(fsp1,fsp2,label,train,test,root):
    Dsp =fsp1*fsp2
    scores = torch.sum(Dsp,1)

    scores = scores.cpu().numpy()
    # print(scores)
    label = label.numpy()
    # print(label)

    FPR,TPR,a_c = roc_curve(label,scores)
    print(len(a_c))

    k = 0
    for i in range(len(TPR)):
        if (TPR[i]>=0.95) and k == 0:
            print("the value of false positive rate at 95% recall is :",FPR[i]*100)
            f = open(root + train + ',test:' + test + '.txt', 'w')
            f.write("the value of false positive rate at 95% recall is :"+ str(FPR[i] * 100)+ "\n")
            k = 1

    # TP = TPR.numpy()
    # RP = FPR.numpy()
    AUCV = auc(FPR,TPR)
    print('AUC = ',AUCV)

    #polt AUC curve
    # plt.figure()
    # plt.plot(FPR,TPR)
    # plt.plot(FPR, FPR, color='red', linewidth=1.0, linestyle='--')
    # # plt.show()
    # savefig('args.batch_size/train:'+train+',test:'+test+'.jpg')

    #save results
    f.write('AUC = '+str(AUCV))
    f.close()










