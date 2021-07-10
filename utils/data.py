import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from PIL import Image

class GradeLoader():

    def __init__(self, root, train=True, obverse=True, transform=None, target_transform=None,
                 download=False, train_split=0.75):

        self.root = root
        grade_list = next(os.walk(root + '/obverse'))[1]
        print('grade_list = ', grade_list)

        self.transform = transform
        self.target_transform = target_transform
        #super(GradeLoader, self).__init__(root, transform=transform,
        #                              target_transform=target_transform)

        self.train = train  # training set or test set

        #if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        self.data = []
        self.targets = []

        if obverse:
            image_dir = root + 'obverse/'
        else:
            image_dir = root + 'reverse/'
        
        dataset = datasets.ImageFolder(image_dir,
                                    transform=transform)

        if self.train:
            train_size = int(train_split * len(dataset))
            test_size = len(dataset) - train_size
           
            traindata, testdata = torch.utils.data.random_split(
                                          dataset, [train_size, test_size])
            trainloader = torch.utils.data.DataLoader(traindata,
                                         batch_size=32, shuffle=True)
            if obverse:
                torch.save(traindata, './datasets/trainset_obv.pt')                
                torch.save(testdata, './datasets/testset_obv.pt')    
            else:
                torch.save(traindata, './datasets/trainset_rev.pt')
                torch.save(testdata, './datasets/testset_rev.pt')
            for img in trainloader:
                images, labels = img
                labels_np = labels.numpy()
                self.data.append(images)
                for el in labels_np:
                    self.targets.append(el)
        else:
            if obverse:
                testdata = torch.load('./datasets/testset_obv.pt')
            else:
                testdata = torch.load('./datasets/testset_rev.pt')

            testloader = torch.utils.data.DataLoader(testdata,
                                         batch_size=32, shuffle=False)

            for img in testloader:
                images, labels = img
                labels_np = labels.numpy()
                self.data.append(images)
                for el in labels_np:
                    self.targets.append(el)

        self.data = np.vstack(self.data).reshape(-1, 3, 255, 255)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        #self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img *= 255.0/img.max()
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

