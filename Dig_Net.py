from __future__ import absolute_import, division
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from torch.autograd import Variable
from Utils import str2bool
from Evaluation_L2 import EvalL2
from Hard_loss import Hardloss
from Dig_net_cov import get_dig_conv
from Dig_loss import Digloss_contrastive,Digloss_triplet
from Get_data import create_train_loaders,create_test_loaders
from tqdm import tqdm
import argparse
import os
import numpy as np
import time
from Losses import loss_HardNet
parser = argparse.ArgumentParser(description='Dig_Net')


parser.add_argument('--data-root', default='/deep_data/BROWN/', type=str, metavar='D',
                    help='the path to your data')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--batch-size', type=int, default=512, metavar='BS',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='BST',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--n-pairs', type=int, default=5000000, metavar='N',
                    help='how many pairs will generate from the dataset')

parser.add_argument('--loss-type', type=str, default="dig_loss_contrastive", metavar='PATH',
                    help='type of training loss: "dig_loss_contrastive","dig_loss_triplet","hard_loss"')

parser.add_argument('--lr', type=float, default=10, metavar='LR',
                    help='same as HardNet')

parser.add_argument('--augmentation', type=str2bool, default=True,
                    help='augmentation of random flip or rotation of 90 deg')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()
cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_pairs * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=1e-4)
    return optimizer

def train(model, train_gen, epoch,optimizer):
        model.train()
        pbar = tqdm(enumerate(train_gen))
        i = 0
        lossdata = 0
        eye = Variable(torch.eye(1024).cuda())
        for batch_idx, data in pbar:
            # if i>=1:
            #     break
            data, target = data
            i = i + 1

            data = data.float().cuda()
            target = target.float().cuda()

            data = Variable(data)
            target = Variable(target)

            imgf1= model(data)
            imgf2 = model(target)
            if args.loss_type == "dig_loss_contrastive":
                loss = Digloss_contrastive(eye,imgf1,imgf2)
            elif args.loss_type == "dig_loss_triplet":
                loss = Digloss_triplet(eye,imgf1,imgf2)
            else:
                loss = loss_HardNet(eye,imgf1,imgf2)



            lossdata = lossdata + loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer)
            if batch_idx % 10 == 0:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_gen.dataset),
                               100. * batch_idx / len(train_gen),
                        lossdata/i))

        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, lossdata/i))

def test(model, test_gen,traindata,testdata,result_root):
        model.eval()
        pbar = tqdm(enumerate(test_gen))
        i = 0
        fsp1 = torch.ones(1, 128).float().cuda()
        fsp2 = torch.ones(1, 128).float().cuda()
        label = torch.ones(1).long()
        for batch_idx, (data_a, data_p, label1) in pbar:
            # if i>=3:
            #     break
            # i = i+1

            data_a = Variable(data_a.float().cuda())
            data_p = Variable(data_p.float().cuda())

            imgf1 = torch.squeeze(model(data_a)).data
            imgf2 = torch.squeeze(model(data_p)).data

            fsp1 = torch.cat((fsp1,imgf1),0)
            fsp2 = torch.cat((fsp2,imgf2),0)
            label = torch.cat((label,label1),0)
            if batch_idx % 10 == 0:
                pbar.set_description(
                               str(100.*batch_idx / len(test_gen)))


        fsp1 = fsp1[1:fsp1.size(0)]
        fsp2 = fsp2[1:fsp2.size(0)]
        label = label[1:label.size(0)]
        print('finish get data, please wait')
        EvalL2(fsp1, fsp2,label, traindata, testdata,result_root)

def main(traindata,testdata1,testdata2):
    if args.augmentation == True:
        saving_root = "dig_net_result_models/"+ args.loss_type + "_" + str(args.epochs) + "_" + str(args.n_pairs) + \
        "_" + str(args.batch_size) + "_" + "aug"
    else:
        saving_root = "dig_net_result_models/"+ args.loss_type + "_" + str(args.epochs) + "_" + str(args.n_pairs) + \
        "_" + str(args.batch_size)

    if not os.path.exists(saving_root):
        os.makedirs(saving_root)

    model = get_dig_conv().cuda()
    optimizer1 = create_optimizer(model.features, args.lr)


    start = args.start_epoch
    end = start + args.epochs

    for epoch in range(start, end):
        train_loader = create_train_loaders(traindata,args.batch_size,args.n_pairs,
                                            args.augmentation,args.data_root)
        # iterate over test loaders and test results
        train( model,train_loader, epoch, optimizer1)
        torch.save(model, saving_root + "/" + args.loss_type + "_" + str(epoch) + "_" + str(args.n_triplets) + \
        "_" + str(args.batch_size) + '.th')

        # release ram
        train_loader=0


    # model = torch.load(saving_root + "/" + args.loss_type + "_" + str(epoch) + "_" + str(args.n_triplets) + \
    #     "_" + str(args.batch_size) + '.th').cuda()
    test(model, create_test_loaders(testdata1,args.batch_size),traindata,testdata1,saving_root)
    time.sleep(10)
    test(model, create_test_loaders(testdata2,args.batch_size),traindata,testdata2,saving_root)




if __name__=='__main__':
    # time.sleep(800)
    main('liberty', 'notredame', 'yosemite')
    main('notredame','yosemite','liberty')
    main('yosemite','liberty','notredame')

