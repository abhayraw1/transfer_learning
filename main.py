import os
import torch
import argparse
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt

from tl import TransferLearner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('data_dir', type=str, help='Dataset Directory.')
    parser.add_argument('--freeze', action='store_true', help='Freeze all but last layer.')
    parser.add_argument('--exp-name', default='exp', help='Experiment name', type=str)
    parser.add_argument('--num-epochs', default=10, help='Number of epochs', type=int)
    args = parser.parse_args()
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), preprocess)
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    params_to_optimize = model.parameters()
    if args.freeze:
        params_to_optimize = model.fc.parameters()
    optimizer = torch.optim.SGD(params_to_optimize, lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    learner = TransferLearner(model, optimizer, loss_fn, exp_lr_scheduler, device, dataloaders)

    stats = learner.learn(num_epochs=args.num_epochs)
    np.savetxt(f'results/{args.exp_name}-stats-val.csv', np.array(stats['val']), delimiter=',')
    np.savetxt(f'results/{args.exp_name}-stats-train.csv', np.array(stats['train']), delimiter=',')
    learner.save_model(f'results/models/{args.exp_name}-model.pkl')
