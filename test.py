# import logging
# import os
# import sys

import torch
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from utils import save_config_file, accuracy, save_checkpoint
# import torch.nn as nn

# import argparse
import torch
# import torch.backends.cudnn as cudnn
# from torchvision import models
# from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
# from simclr import SimCLR


from torchvision import transforms, datasets

BATCH_SIZE = 256
test_dataset = datasets.CIFAR10(root='./testdata/',
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                            download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=False,
                                          num_workers=4)

model = ResNetSimCLR(base_model='resnet18', out_dim=128)
model.load_state_dict(torch.load('/Users/zixiwang/Downloads/checkpoint_0100.pth.tar', map_location=torch.device('cpu'))['state_dict'])


correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        prediction, features = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(prediction.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))