# RAL-Net

##This code is the ACCV2018 paper [Robust Angular Local Descriptor Learning](http://igm.univ-mlv.fr/~cwang/papers/ACCV2018_DescriptorLearning.pdf)

##Enviromental built: 

pytorch 0.4, opencv 3.4

## To replicate the result of this paper:

python RAL_Net.py --data-root=/mnt/Brain/wug/Brown_data/ --epochs=10 --batch-size=512 --n-pairs=5000000 --loss-type=RAL_loss --lr=10 --augmentation=True

## Result:

# Performance comparison on Brown daraset, lower score and perform better

<img src="https://github.com/xuyanwu/RAL-Net/blob/master/Result/BROWN.PNG" width="700">

# Performance comparison on Hpatches daraset, Higher score and perform better

<img src="https://github.com/xuyanwu/RAL-Net/blob/master/Result/Hpatches.PNG" width="700">

# Performance comparison on Wxbs daraset, Higher score and perform better

<img src="https://github.com/xuyanwu/RAL-Net/blob/master/Result/Wxbs.PNG" width="700">
