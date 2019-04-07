# This is a command line function for training a CNN using a folder ofimages supplied.
#
# Load libraries to be use_device# Imports Numpy, PyTorch,
# TorchVision and other supporting packages
#

import numpy as np

import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import time
import argparse
from collections import OrderedDict

# Parsing parameters from command line
#
#
parser = argparse.ArgumentParser(
    description = 'This is my little flower classifier program.',
    add_help = True,
    epilog = 'This is the end of the program.')

# mandatory parameters
parser.add_argument('data_dir',
                    help = 'The data directory. (required)')

# optional parameters
parser.add_argument('--arch', action='store',
                    default = 'vgg19',
                    dest = 'use_arch',
                    help = 'Set directory for checkpoint to use.')

parser.add_argument('--use_checkpoint_dir', action='store',
                    default = '', # is blank if none specified
                    dest = 'use_checkpoint_dir',
                    help = 'Set directory for checkpoint to use.')

parser.add_argument('--learn_rate', action='store', type = float,
                    default = 0.01, # default learnrate is 0.01
                    dest = 'use_learn_rate',
                    help = 'Specify the default learn rate for the CNN.')

parser.add_argument('--hidden_units', action='store', type = int,
                    default = 4096, # default is 4086 units
                    dest = 'use_hidden_units',
                    help = 'Specify number of hidden units in the classifier.')

parser.add_argument('--epochs', action = 'store', type = int,
                    default = 1, # default is 1 epoch
                    dest = 'use_epochs',
                    help = 'Specify number of epochs to train the CNN.')

parser.add_argument('--device', action='store',
                    default = 'cpu',
                    dest = 'request_device',
                    help = 'Choose between cpu and gpu.')

user_input = parser.parse_args()

# Provide confirmation of options parsed
print('\nRunning train.py with the following parameters:')
print('Data directory location:', user_input.data_dir)
print('CNN architecture:', user_input.use_arch)
print('Checkpoint filename:', user_input.use_checkpoint_dir)
print('Learn rate:', user_input.use_learn_rate)
print('Hidden units:', user_input.use_hidden_units)
print('Epochs:', user_input.use_epochs)
print('Device:', user_input.request_device)


# Check to see whether using gpu or cpu
# assumption: CPU is always available
# only use GPU if GPU is available and the command line requests it
if (user_input.request_device == 'gpu') and (torch.cuda.is_available()):
    use_device = 'cuda:0'
else:
    use_device = 'cpu'

    
# Determine which pre-trained model architecture to use
# based on option entered
if user_input.use_arch == 'vgg16':
    model = models.vgg16(pretrained = True)
elif user_input.use_arch == 'vgg19':
    model = models.vgg19(pretrained = True)
elif user_input.use_arch == 'vgg13':
    model = models.vgg13(pretrained = True)
else:
    print('Model architecture {} unsupported: using vgg19 instead.'.format(model_arch))
    model = models.vgg19(pretrained = True)


# Train the model using data directory and 
# hyperparameters supplied
    
start_time = time.time()
print('\nLoading data for training ...')

# Set up data directories
data_dir = user_input.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms
local_batch_size = 8     # Allow the use of different batch sizes 

# random crop between 50% and 90%
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(256, [0.5, 0.9]), # scaling
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                       ])
# keeping image size to 224, as original image sizes vary
data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])
# Load train set with ImageFolder and define the dataloader
trainset = torchvision.datasets.ImageFolder(train_dir,
                                            transform = train_transforms) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size = local_batch_size,
                                          shuffle = True)

# Load valid set with ImageFolder and define the dataloader
validset = torchvision.datasets.ImageFolder(valid_dir,
                                            transform = data_transforms)
validloader = torch.utils.data.DataLoader(validset, batch_size = local_batch_size,
                                          shuffle = True)

# Check imported items 
print('There are {} items in the trainset'.format(len(trainset)))
print('There are {} items in the trainloader'.format(len(trainloader)))
print('There are {} items in the valid'.format(len(validset)))
print('There are {} items in the validloader'.format(len(validloader)))


# Freeze parameters so we don't backprop through them. 
# Done before assigning the new classifier layer.
for param in model.parameters():
    param.requires_grad = False 

# Create the classifier to replace the pre-trained one
dropout_p = 0.5 # 50% chance drop out

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, user_input.use_hidden_units)), # model has 25088 in_features
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Dropout(p = dropout_p)), 
                          ('fc2', nn.Linear(user_input.use_hidden_units, 102)), # model has 102 out_features
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
# Assign classifer to model
model.classifier = classifier


# Train the newly defined classifier
print('\nTraining model ... now do some stretching exercises ...')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=user_input.use_learn_rate)
model.to(use_device)

# Set up model for training
training_steps = 0
print_every = 50 # batches of images

print('There are {} batches of images to train.'.format(len(trainloader)))

for epoch in range(user_input.use_epochs):
    model.train()
    training_loss = 0 

    # iterate images
    for images, labels in iter(trainloader):
        training_steps += 1

        # move images and labels to the chosen device
        images, labels = images.to(use_device), labels.to(use_device)

        # set parameter graidents to zero
        optimizer.zero_grad()

        # feed forward!
        outputs = model.forward(images)

        # backprop!
        loss = criterion(outputs, labels) 
        loss.backward() # feedback loss backword through the network
        optimizer.step()

        training_loss += loss.item()

        if training_steps % print_every == 0:
            # Show training progress
            print('Epoch: {}/{} ...'.format(epoch+1, user_input.use_epochs), 
                  'Training loss: {:.4f}'.format(training_loss/print_every))
            training_loss = 0
    print('End of epoch', epoch+1)

    # Run validation loss and accuracy test
    print('\nCalculating validation loss and accuracy for epoch {}...'.format(epoch+1))
    
    # These values are re-set for each epoch
    is_correct_total = 0
    total_test_samples = len(validloader) * local_batch_size
    n_samples_evaluated = 0

    valid_loss = 0.0 
    valid_steps = 0 
    
    with torch.no_grad():
        model.eval() # put model in evaluation mode

        # Run validation 
        for valid_images, valid_labels in validloader:
            valid_images, valid_labels = valid_images.to(use_device), valid_labels.to(use_device)

            # Run forward pass
            outputs = model(valid_images)
            n_samples_evaluated += local_batch_size

            # Calculate validation loss
            loss = criterion(outputs, valid_labels)
            valid_loss += loss.item() 

            # Calculate accuracy
            _, pred_labels = torch.max(outputs, dim=1)
            is_correct = (valid_labels == pred_labels).sum().item()
            is_correct_total += is_correct
            accuracy = is_correct_total/n_samples_evaluated

            valid_steps += 1

    print('Validation loss: {:.2f}  Validation accuracy: {:.2%}\n'.format(valid_loss/valid_steps,
                                                                          accuracy))

# Calculate how long it takes to train and validate
time_taken = time.time() - start_time
print('Training and validation took {:.0f}m {:.0f}s\n'.format(time_taken // 60, time_taken % 60))

# Assgine class to index to model

# Function to save checkpoint
#
# 
if user_input.use_checkpoint_dir != '':
    model.class_to_idx = trainset.class_to_idx
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    print('Saving model:\n', model, '\n')
    print('with the state_dict keys:\n', model.state_dict().keys(), '\n')

    checkpoint = {'model_arch': user_input.use_arch,
                  'hidden_units': user_input.use_hidden_units,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()
                 }

    print('Saving to file:', user_input.use_checkpoint_dir, '\n')
    torch.save(checkpoint, user_input.use_checkpoint_dir)
else:
    print('Checkpoint not saved as file directory not supplied.')
    
# Calculate how long it takes to run the whole script
time_taken = time.time() - start_time
print('Total time taken {:.0f}m {:.0f}s\n'.format(time_taken // 60, time_taken % 60))
print('End.')