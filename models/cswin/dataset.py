import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import PIL

DATA_PATH = os.environ.get('MVIT_PATH', 'yourpath')


class ABAP():
    def __init__(self, dataset, DATA_PATH, trainsize, testsize, aug=True):
        if dataset in ['abap', 'gsv', 'gsv_unlabel']:
            self.dataset = dataset
        else:
            raise NotImplementedError
        
        if self.dataset == 'gsv':
            print('****gsv*** ------ Test only')
            self.DATA_PATH = DATA_PATH.replace('abap', 'gsv')
        elif self.dataset == 'gsv_unlabel':
            print('****gsv_unlabel*** ------ Test only')
            self.DATA_PATH = DATA_PATH.replace('abap', 'gsv_unlabel')
        else:
            if aug:
                self.DATA_PATH = DATA_PATH
            else:
                self.DATA_PATH = DATA_PATH.replace('abap', 'unbalanced')

        self.trainsize = trainsize
        self.testsize = testsize

    def transforms(self):
        train_transform  = transforms.Compose([
            transforms.RandomResizedCrop(
                    (224, 224),
                    scale=(0.08, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = train_transform
        return train_transform, test_transform

    def prepare_data(self):
        train_transform, test_transform = self.transforms()
        traindir = os.path.join(self.DATA_PATH, 'train')
        valdir = os.path.join(self.DATA_PATH, 'test')
        self.trainset = datasets.ImageFolder(traindir, transform=train_transform)
        self.testset = datasets.ImageFolder(valdir, transform=test_transform)

    def samplers(self):
        train_batch_sampler = torch.utils.data.sampler.RandomSampler(self.trainset)
        test_batch_sampler = torch.utils.data.sampler.RandomSampler(self.testset)
        trainsampler = torch.utils.data.BatchSampler(
            train_batch_sampler,
            batch_size=self.trainsize, drop_last=False,
        )
        testsampler = torch.utils.data.BatchSampler(
            test_batch_sampler,
            batch_size=self.testsize, drop_last=False,
        )
        return trainsampler, testsampler


    def dataloaders(self):
        trainsampler, testsampler = self.samplers()
  
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=0,
            pin_memory=True,
            batch_sampler=trainsampler,
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=0,
            pin_memory=True,
            batch_sampler=testsampler,
        )

        return train_loader, test_loader
    

if __name__ == '__main__':
    abap = ABAP(dataset='gsv', DATA_PATH=DATA_PATH, trainsize=2, testsize=2)
    abap.prepare_data()
    train, test = abap.dataloaders()
    print(train)
    for step, (x, y) in enumerate(train):
        print(np.shape(x), y)
        break
    for step, (x, y) in enumerate(test):
        print(np.shape(x), y)
        break
    # train_transform  = transforms.Compose([
    #         transforms.RandomResizedCrop(
    #                 224,
    #                 scale=(0.08, 1.0),
    #                 interpolation=PIL.Image.BICUBIC,
    #             ),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ])
    # traindir = os.path.join(MVIT_PATH, 'train')
    # trainset = datasets.ImageFolder(traindir, transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(
    #         trainset,
    #         num_workers=0,
    #         pin_memory=True,
    #         batch_size=3,
    #         shuffle=True
    #     )
    # for step, (x, y) in enumerate(train_loader):

    #     print(np.shape(x), y)
    
