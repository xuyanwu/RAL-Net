from __future__ import division, print_function
from copy import deepcopy
import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import random
from Utils import cv2_scale, np_reshape
from tqdm import tqdm



mean_image = 0.443728476019
std_image = 0.20197947209
image_Size = 32
kwargs = {'num_workers': 4, 'pin_memory': True}


class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None,N_pairs = None, aug = False,
                 load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.N_pairs = N_pairs
        self.batch_size = batch_size
        self.aug = aug

        if self.train:
            print('Generating {} triplets'.format(self.N_pairs))
            self.triplets = self.generate_triplets(self.labels, self.N_pairs,self.batch_size)

    @staticmethod
    def generate_triplets(labels, num_triplets,batch_size):
        # print (len(labels))
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p = self.data[t[0]], self.data[t[1]]

        img_a = transform_img(a)
        img_p = transform_img(p)

        if self.aug:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
        return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)





def create_train_loaders(train_data,batch_size,n_pairs,augmentation,dataroot):


    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((mean_image,), (std_image,))])

    train_loader = torch.utils.data.DataLoader(
            TripletPhotoTour(train=True,
                             load_random_triplets = False,
                             batch_size=batch_size,
                             N_pairs=n_pairs,
                             aug = augmentation,
                             root=dataroot,
                             name=train_data,
                             download=True,
                             transform=transform),
                             batch_size=batch_size,
                             shuffle=True, **kwargs)

    return train_loader

def create_test_loaders(test_data,batch_size):


    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((mean_image,), (std_image,))])

    test_loader =torch.utils.data.DataLoader(
                         TripletPhotoTour(train=False,
                                          batch_size=batch_size,
                                          root=dataroot,
                                          name=test_data,
                                          download=True,
                                          transform=transform),
                         batch_size=batch_size,
                         shuffle=False, **kwargs)
    return test_loader

