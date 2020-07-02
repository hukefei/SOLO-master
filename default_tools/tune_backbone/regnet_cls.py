#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from mmdet.models import RegNet


class RegNet_cls(RegNet):
    def __init__(self,
                 arch='regnetx_800mf',
                 base_channels=32,
                 num_classes=4,
                 pretrained=None):
        super().__init__(arch, base_channels=base_channels)
        self.num_classes = num_classes
        self.avg_pool = nn.AvgPool2d((7, 7), (1, 1))
        self.last_linear = nn.Linear(672, num_classes)
        self.init_weights(pretrained=pretrained)

    def forward(self, inputs):
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


if __name__ == '__main__':
    model = RegNet_cls(arch='regnetx_800mf', base_channels=32, num_classes=4,
                       pretrained='/versa/dyy/pretrained_models/RegNetX-800MF_dds_8gpu.pth')
    # print(model)
    # for i, (k, v) in enumerate(model.state_dict().items()):
    #     print(i, k, v.shape)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.cuda()
    # model = nn.DataParallel(model, device_ids=[0])

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '/versa/dyy/coco'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x+'_gt'),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=512, num_workers=4,
                                                  shuffle=True if x == 'train' else False)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # class_names = image_datasets['train'].
    # print(class_names)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-4)  # T_max means 1/2 cosine period

    num_epochs = 50

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(),
                   '/versa/dyy/SOLO/work_dirs/backbone/reg800/tuned-{}-{}.pth'.format(epoch, epoch_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.cpu().state_dict(),
               '/versa/dyy/SOLO/work_dirs/backbone/reg800/tuned-best-{}.pth'.format(best_acc))
