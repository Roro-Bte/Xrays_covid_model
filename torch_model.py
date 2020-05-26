import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = x.view(x.shape[0], -1)
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


def validation(model, test_loader):
    accuracy = 0
    accuracy2 = 0

    with torch.no_grad():
        model.eval()
        
        for images, labels in test_loader:
#             #move tensors to GPU if CUDA is available
#             if train_on_gpu:
#                 data, target = data.cuda(), target.cuda()
                
            log_ps = model(images)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape) # change this if I want to compare two different classes.
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            # calculate accuracy between normal and sick only (2 classes):
            originais = labels.view(*top_class.shape)
            mask = originais[:,0]>=1
            originais[mask,:] = 1
            new_pred = top_class
            mask = new_pred[:,0]>=1
            new_pred[mask,:] = 1
            equals2 = new_pred == originais
            accuracy2 += torch.mean(equals2.type(torch.FloatTensor))
        
    print("Test Accuracy for 7 classes: {:.1f} %".format((accuracy /len(test_loader)).item()*100))
    print("Test Accuracy for 2 classes: {:.1f} %".format((accuracy2/len(test_loader)).item()*100))


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs=5):
    
    steps = 0
    train_losses, test_losses = [], []
    for e in range(epochs):
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