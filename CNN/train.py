from __future__ import print_function # future proof
import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm, trange


# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models




# Load the training data 
def _get_loader(batch_size, data_dir, editions, valid_size):
    """Loads data to a train and validation loader
    """
    
    print("Get data loader.")
    dataset = datasets.ImageFolder(data_dir, transform=editions)
    print("Classes and their index:")
    print(dataset.class_to_idx)
    
    # obtain training indices that will be used for validation
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=valid_sampler)
    
    return train_loader, valid_loader


# Provided train function
def train(model, train_loader, valid_loader, epochs, optimizer, criterion, device, model_dir, model_name, output_dir):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    valid_loader - The validation loader that prevents overfitting
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    criterion    - The loss function used for training. 
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    print("Starting training with {} epochs".format(epochs))

    train_losses, valid_losses = [], []
    valid_loss_min = np.Inf # track change in validation los
    for e in range(epochs):
        
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        batch= 1
        for images, labels in train_loader:
#             print("working on training batch: {}".format(batch) )
            # move tensors to GPU if is available
            images, labels = images.to(device), labels.to(device)
            model.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images)
            # calculate the batch loss
            loss = criterion(output, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*images.size(0)
            batch += 1
            
        ######################    
        # validate the model #
        ######################
        
        accuracy = 0
        model.eval()
        batch= 1
        for images, labels in valid_loader:
#             print("working on validation batch: {}".format(batch) )
            # move tensors to GPU if is available
            images, labels = images.to(device), labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images)
            # calculate the batch loss
            loss = criterion(output, labels)
            # get highest probability. It returns a tuple with the probability and its index
            top_p, top_class = output.topk(1, dim=1)
            # transform labels to size (batch, 1) to compare with top class
            equals = top_class == labels.view(*top_class.shape)
            # update accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            valid_loss += loss.item()*images.size(0)
            batch += 1
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        

        # Save values for later analysis
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Print stats
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Validation Loss: {:.3f}.. ".format(valid_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)),

             )
            
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            save_model(model, model_dir)
            valid_loss_min = valid_loss
            

    # Create df of train and validation losses
    summary = pd.DataFrame({'epoch': list(range(1,len(train_losses)+1)),
              'train_loss' : train_losses,
              'valid_loss' : valid_losses              
             }
            )

    # export as csv
    csv_name = model_name + '_summary.csv'
#     os.makedirs('model_data', exist_ok=True)
    summary.to_csv(os.path.join(output_dir, csv_name))

# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")

    checkpoint = {'n_input': args.n_input,
                  'n_classes': args.n_classes,
                  'state_dict': model.cpu().state_dict()}

    name = 'cnn.pth'
    torch.save(checkpoint, os.path.join(model_dir, name))
    
    

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--test_dir', type=str, default=os.environ['SM_CHANNEL_TESTING'])
               
    # Model parameters
    parser.add_argument('--n_input', type=int, default=4096, metavar='IN',
                        help='number of input features to model (default: 2)')
    parser.add_argument('--n_classes', type=int, default=4, metavar='OUT',
                        help='output dim of model (default: 1)')  
    parser.add_argument('--freeze_all', type=str, default='True', metavar='FREEZE',
                        help='Freeze training for all "features" layers (default: True)')   
    parser.add_argument('--model_name', type=str, default='model_default_name', metavar='NAME',
                        help='model name (default: model_default_name)')   
    
    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--valid_size', type=float, default=0.15, metavar='VAL',
                        help='validation size from the training set')



    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Compose transforms
    editions = transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.RandomRotation(4),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                        
                                  ])
    
    
    # get train loader
    train_loader, valid_loader = _get_loader(args.batch_size, args.train_dir, editions, args.valid_size)
    print("Num of batches in train data: ",len(train_loader))
    print("Num of batches in validation data: ",len(valid_loader))
    
    ## Build the model
    model = models.vgg16(pretrained=True)
    
    # Freeze training for all "features" layers
    if args.freeze_all == 'True':
        print('Freezing training for all "features" layers')
        for param in model.features.parameters():
            param.requires_grad = False
    elif args.freeze_all == 'False':
        print('Unfreezing training for all "features" layers')
    
    # Change last layer
    model.classifier[6] = nn.Linear(args.n_input, args.n_classes)
    
    ## Define an optimizer and loss function for training
    print("Learning rate: {}".format(args.lr) )
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Move to gpu 
    model.to(device)
    
    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, train_loader, valid_loader, args.epochs, optimizer, criterion, device, args.model_dir, 
          args.model_name, args.output_dir)