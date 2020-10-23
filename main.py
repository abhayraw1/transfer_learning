import os
import time
import torch
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt

from tl import TransferLearner


if __name__ == '__main__':
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), preprocess)
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=1)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ALL PARAMS BEING OPTIMIZED
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    learner = TransferLearner(model, optimizer, loss_fn, exp_lr_scheduler, device, dataloaders)

    # learner.learn(num_epochs=5)
    loss = 0.
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss += learner.train_on_batch(inputs, labels)
    print(loss)

    loss = 0.
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss += learner.eval_on_batch(inputs, labels)
        print(loss)
