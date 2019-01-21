# RAL-Net

1.This code is the ACCV2018 paper [Robust Angular Local Descriptor Learning](http://igm.univ-mlv.fr/~cwang/papers/ACCV2018_DescriptorLearning.pdf)

2.Enviromental built: pytorch 0.4

3.To replicate the result of this paper:

python RAL_Net.py --data-root=/mnt/Brain/wug/Brown_data/ --epochs=10 --batch-size=512 --n-pairs=5000000 --loss-type=RAL_loss --lr=10 --augmentation=True

Result:

