from __future__ import print_function # future proof
import argparse
import sys
import os
import json
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

# import model
from model import Network


def model_fn(model_dir):
    
    print("Loading model.")
    model_path = os.path.join(model_dir, 'model_info.pth')

    # Determine the device and construct the model.
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu" # change in AWS
    
    checkpoint = torch.load(model_path)
    model = Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    print(model)
    return model.to(device)


# Load the training data 
def _get_train_loader(batch_size, data_dir, editions):
    print("Get data loader.")
    train_data = datasets.ImageFolder(data_dir, transform=editions) 
    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Provided train function
def train(model, train_loader, test_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    print("Starting training with {} epochs".format(epochs))
    steps = 0
    train_losses, test_losses = [], []
    for e in trange(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            test_loss = 0
            accuracy = 0
            accuracy2 = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1) # get highest probability. It returns a tuple with the probability and its index
                    equals = top_class == labels.view(*top_class.shape) # transform labels to size (batch, 1) to compare with top class
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
                    # calculate accuracy only between normal lung and sick (2 classes):
                    originais = labels.view(*top_class.shape)
                    mask = originais[:,0]>=1
                    originais[mask,:] = 1
                    new_pred = top_class
                    mask = new_pred[:,0]>=1
                    new_pred[mask,:] = 1
                    equals2 = new_pred == originais
                    accuracy2 += torch.mean(equals2.type(torch.FloatTensor))

            model.train()

            # Save values to show on a graph later
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))

            # Print stats
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                  "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)),
                  "Test Accuracy2: {:.3f}".format(accuracy2/len(test_loader))
                 )
            
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)

    # save trained model, after all epochs
    save_model(model, args.model_dir)


# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")

#     torch.save(model.cpu().state_dict(), path)
    
    checkpoint = {'input_size': args.input_dim,
                  'output_size': args.output_dim,
                  'hidden_layers': args.hidden_dims, #[each.out_features for each in model.hidden_layers],
                  'droprate': args.drop,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, path)
    
def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_param.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
                    'input_dim': args.input_dim,
                    'output_dim': args.output_dim,
                    'hidden_layers': args.hidden_dims, #[each.out_features for each in model.hidden_layers],
                    'droprate': args.drop
                     }
        torch.save(model_info, f)
    
    



if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
#     parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
#     parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--test_dir', type=str, default='data/test')
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  
    # Add args for the three model parameters: input_dim, hidden_dim, output_dim
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=196608, metavar='IN',
                        help='number of input features to model (default: 2)')
    parser.add_argument('--output_dim', type=int, default=7, metavar='OUT',
                        help='output dim of model (default: 1)')
    parser.add_argument('--hidden_dims', type=list, default=[512, 256, 128, 64], metavar='H',
                        help='hidden dimensions of model (default: 512,256,128,64)')
    parser.add_argument('--drop', type=float, default=0.1, metavar='DROP',
                        help='dropout of model (default: 0.1)')
    
    args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu" # change in AWS
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        

    # Compose transforms
    editions = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(256),
    #                                transforms.ColorJitter(),
    #                                transforms.RandomGrayscale(),
                                   transforms.RandomRotation(4),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # new thing                            
                                  ])
    
    
    # get train loader
    train_loader = _get_train_loader(args.batch_size, args.data_dir, editions) # data_dir from above..
    print("Num of batches in train data: ",len(train_loader))
    test_loader = _get_train_loader(args.batch_size, args.test_dir, editions)
    print("Num of batches in test data: ",len(test_loader))
    
    ## Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = Network(args.input_dim, args.output_dim, args.hidden_dims, args.drop).to(device)
    
    # Given: save the parameters used to construct the model
    save_model_params(model, args.model_dir)

    ## Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    
    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, train_loader, test_loader, args.epochs, optimizer, criterion, device)