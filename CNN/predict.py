import argparse
import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from io import StringIO
from six import BytesIO

# import model
from torchvision import datasets, transforms, models

# accepts and returns numpy data
CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    name = 'cnn.pth'
    checkpoint = torch.load(os.path.join(model_dir, name))
    model = models.vgg16(pretrained=True)
    # Freeze training for all "features" layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    model.classifier[6] = nn.Linear(checkpoint['n_input'], checkpoint['n_classes'])
    model.load_state_dict(checkpoint['state_dict'])
    
    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# def input_fn(serialized_input_data, content_type):
#     print('Deserializing the input data.')
#     if content_type == CONTENT_TYPE:
#         stream = BytesIO(serialized_input_data)
#         return np.load(stream)
#     raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# def output_fn(prediction_output, accept):
#     print('Serializing the generated output.')
#     if accept == CONTENT_TYPE:
#         buffer = BytesIO()
#         np.save(buffer, prediction_output)
#         return buffer.getvalue(), accept
#     raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
#     data = torch.from_numpy(input_data.astype('float32'))
    data = input_data
    data = data.to(device)

    # Put model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data.
    out = model(data)
    # The variable `result` should be a numpy array; a single value 0-1
#     result = out.cpu().detach().numpy()

    return out