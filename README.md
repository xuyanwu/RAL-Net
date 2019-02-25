## RAL-Net

This code is the ACCV2018 paper [Robust Angular Local Descriptor Learning](https://arxiv.org/pdf/1901.07076.pdf)

## Enviromental built: 

Please install python 3.6, pytorch 0.4.1, opencv 3.4

pip install python==3.6

conda install pytorch=0.4.1 cuda90 -c pytorch


## To replicate the result of this paper:

python RAL_Net.py --data-root=<<your data root>> --epochs=10 --batch-size=512 --n-pairs=5000000 --loss-type=RAL_loss --lr=10 --augmentation=True

## Result:

### Performance comparison on Brown daraset, lower score and perform better

<img src="https://github.com/xuyanwu/RAL-Net/blob/master/Result/BROWN.PNG" width="600">

### Performance comparison on Hpatches daraset, Higher score and perform better

<img src="https://github.com/xuyanwu/RAL-Net/blob/master/Result/Hpatches.PNG" width="600">

### Performance comparison on Wxbs daraset, Higher score and perform better

<img src="https://github.com/xuyanwu/RAL-Net/blob/master/Result/Wxbs.PNG" width="600">
