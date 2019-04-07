# This is a command line function for predicting a image supplied.
# In return the function will provide the flower name and the probability
# of the name.

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

from PIL import Image

import matplotlib.pyplot as plt

import json

# Load checkpoint using supplied file path
#
#
def load_checkpoint(file_path, use_this_device):

    print('- Loading checkpoint from file:',file_path)
    print('- Using device:', use_this_device)

    checkpoint = torch.load(file_path, map_location = use_this_device)

    model_arch = checkpoint['model_arch']

    if model_arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        print('- Model architecture {} unsupported: using vgg19 instead.'.format(model_arch))
        model = models.vgg19(pretrained = True)

    # Freeze parameters!
    for param in model.parameters():
        param.requires_grad = False

    # Recreate the classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, checkpoint['hidden_units'])), # model has 25088 in_features
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Dropout(p = 0.5)), # this is defined in train.py
                          ('fc2', nn.Linear(checkpoint['hidden_units'], 102)), # model has 102 out_features
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load class to idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model


# Convert PIL image in image_path into a PyTorch usable
# Numpy array
#
def process_image(image_path):
    print('- Processing image...')
    image = Image.open(image_path)

    # Resize image to 256 shortest side
    width, height = image.size
    # print('Image original size:', image.size)
    if width >= height:
        image = image.resize((int(width/height*256), 256), resample = 0)
    else:
        image = image.resize((256, int(height/width*256)), resample = 0)
    # print('New image size:', image.size)

    # Crop the centre 224x224
    dim224 = 224
    width, height = image.size
    if (width or height) > dim224:
        left = (width - dim224)/2
        top = (height - dim224)/2
        right = (width + dim224)/2
        bottom = (height + dim224)/2

        image = image.crop((left, top, right, bottom))
    else:
        image = image.resize((224, 224)) # if the supplied image is smaller than 224

    # Nornalise for PyTorch
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std

    # Transpose into the correct sequence of dimensions as PyTorch expect
    image = image.transpose((2, 0, 1))

    return image # as a Numpy array


# Display thte top n classes
#
#

def predict_image(this_image, model_to_use, n_predictions, cat_to_name, use_device):
    
    print('\nRunning prediction...')

    # Put model in evaluation mode
    model_to_use.eval()
    model_to_use.to(use_device) # move model to chosen device

    # Convert image to Pytorch float tensor
    if use_device == 'cpu':
        image_tensor = torch.from_numpy(this_image).type(torch.FloatTensor)
    else:
        image_tensor = torch.from_numpy(this_image).type(torch.cuda.FloatTensor) # sends this to CPU
    image_tensor = image_tensor.view(1, 3, 224, 224) # add batch to front of tensor

    # Predict!
    with torch.no_grad():
        output = model_to_use(image_tensor)
        ps = torch.exp(output)
        top_probabilities, top_labels = ps.topk(n_predictions, dim = 1)
        if use_device != 'cpu': # return the tensor from GPU to CPU
            top_probabilities = top_probabilities.cpu()
            top_labels = top_labels.cpu()
        top_probabilities = top_probabilities.numpy().squeeze()
        top_labels = top_labels.numpy().squeeze()        
    
    # convert class to indices to indices to class, 
    idx_to_class = {val: key for key, val in model_to_use.class_to_idx.items()}

    # Show 1 prediction, or a list of predictions
    if n_predictions == 1:
        # show a single prediction
        top_class = idx_to_class[top_labels.item(0)]
        named_class = cat_to_name[str(top_class)]
        top_probability = top_probabilities.item(0)
        print('The image is likely to be _{}_ with probability of {:.4f}'.format(named_class,
                                                                              top_probability))       
    else:
        # show k predictions
        # convert labels to classes
        top_classes = [idx_to_class[label] for label in top_labels]
        named_classes = [cat_to_name[str(top_classes_)] for top_classes_ in top_classes]
        predicted_classes = pd.DataFrame({'Probability':top_probabilities}, named_classes)
        print(predicted_classes)
        
    return None


# Parsing parameters from command line
#
#
import argparse

parser = argparse.ArgumentParser(
    description = 'This is my little flower class prediction program.',
    add_help = True,
    epilog = 'This is the end of the program.')

# mandatory parameters
parser.add_argument('user_filename',
                    help = 'The filename for prediction. (required)')

parser.add_argument('use_checkpoint_dir', action='store',
                    help = 'Set directory for checkpoint to use.')

# optional parameters
parser.add_argument('--top_k', action='store',
                    type = int,
                    default = 1, # default is to predict 1
                    dest = 'use_topk',
                    help = 'Set top KKK most likely cases.')

parser.add_argument('--category_names', action = 'store',
                    default = '',
                    dest = 'use_cat_filename',
                    help = 'Provide category names mapping file.')

parser.add_argument('--device', action='store',
                    default = 'cpu',
                    dest = 'request_device',
                    help = 'Choose between cpu and gpu.')

user_input = parser.parse_args()



# Provide confirmation of options parsed
#
#
print('\nRunning predict.py')
print('- Using filename:', user_input.user_filename)
print('- Using checkpoint file:', user_input.use_checkpoint_dir)

# Check if category names have been supplied
if user_input.use_cat_filename != '':
    print('- Using category names stored in file:', user_input.use_cat_filename)
    with open(user_input.use_cat_filename, 'r') as f:
        cat_to_name = json.load(f)     
else:
    print('- Using default category names.')
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)    
      
# Check to see whether using gpu or cpu
# On usef GPU if available
if (user_input.request_device == 'gpu'):
    if torch.cuda.is_available():
        print('- GPU available for use.')
        use_device = 'cuda:0'
    else:
        print('- GPU unavailable for use: using CPU instead.')
        use_device = 'cpu'
else:
    use_device = 'cpu'

# Decide whether to predict one or K classes
if user_input.use_topk == 1:
    prediction_reqd = 1
else:
    prediction_reqd = user_input.use_topk

# Process image 
processed_image = process_image(user_input.user_filename)

# Load checkpoint and predict image
predict_model = load_checkpoint(user_input.use_checkpoint_dir, use_device)
    
predict_image(processed_image,
              predict_model,
              prediction_reqd,
              cat_to_name,
              use_device)

print('\nEnd.')