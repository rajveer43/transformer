import copy

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

from vision.vit import ViT

import matplotlib.pyplot as plt


def plot_accuracy(accuracy):
    x = list(range(len(accuracy['train'])))

    fig = plt.figure()
    plt.title('Accuracy over Epochs', fontsize=12, fontweight='bold')

    font = {'color': 'black', 'weight': 'normal', 'size': 13}

    plt.xlabel('Epoch(s)', fontdict=font)
    plt.ylabel('Accuracy(%)', fontdict=font)

    plt.plot(x, accuracy['train'], color='steelblue', label='Training Accuracy')
    plt.plot(x, accuracy['val'], color='darkred', label='Validation Accuracy')

    plt.legend(prop={'size': 10})
    plt.show()


BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 1e-3

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
        'train': torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=data_transforms['train']
        ),
        'val': torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=data_transforms['val']
        )
    }

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ViT(image_res=224, patch_res=16, num_classes=num_classes) # vit-base.

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

model_accuracy = {'train': [], 'val': []}

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

prev_acc = 0.0
for epoch in range(NUM_EPOCHS):
    print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
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
            inputs = inputs.to(device)
            labels = labels.to(device)

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

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        model_accuracy[phase].append(epoch_acc * 100)

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()
print('Best val Acc: {:4f}'.format(best_acc))

PATH = './pretrained_models/' + model + '.pth'
torch.save(best_model_wts, PATH)

plot_accuracy(model_accuracy)